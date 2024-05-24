# Define the dataset and dataloader
input_dir = 'coco_preprocessed/inputs_small'
output_dir = 'coco_preprocessed/outputs_small'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CustomDataset(input_dir, output_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Initialize models and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = UNetGenerator().to(device)
discriminator = PatchDiscriminator().to(device)
perceptual_loss = PerceptualLoss().to(device)

# Adjust learning rates for specific layers
def get_optimizers(generator, discriminator):
    # Different learning rates for self-attention layers
    attention_params = [p for n, p in generator.named_parameters() if 'query_conv' in n or 'key_conv' in n or 'value_conv' in n]
    other_params = [p for n, p in generator.named_parameters() if 'query_conv' not in n and 'key_conv' not in n and 'value_conv' not in n]

    g_optimizer = optim.Adam([
        {'params': attention_params, 'lr': 0.000005},
        {'params': other_params, 'lr': 0.0002}
    ], betas=(0.5, 0.999))

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))  # Lower LR for discriminator
    return g_optimizer, d_optimizer

# Call the function to get optimizers
g_optimizer, d_optimizer = get_optimizers(generator, discriminator)

# Function to train the model
def train_model(generator, discriminator, perceptual_loss, dataloader, num_epochs, device, display_interval=1):
    adversarial_loss = LSGANLoss()
    l1_loss = nn.L1Loss()

    # Adjust the loss weights to balance training
    lambda_adv = .1
    lambda_l1 = .05
    lambda_perceptual = 0.05

    # Learning rate scheduler
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.1)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in progress_bar:
            real_images = data['real'].to(device)
            input_images = data['input'].to(device)
            batch_size = real_images.size(0)
            
            # Train Generator
            g_optimizer.zero_grad()
            
            generated_images = generator(input_images)
            discriminator_output_fake = discriminator(generated_images)

            # Generate target tensors with correct size
            valid = torch.ones_like(discriminator_output_fake, requires_grad=False).to(device)
            fake = torch.zeros_like(discriminator_output_fake, requires_grad=False).to(device)
            
            g_loss_adv = adversarial_loss(discriminator_output_fake, valid)
            
            # Resize real images to match the generated images for L1 loss
            resized_real_images = nn.functional.interpolate(real_images, size=(generated_images.size(2), generated_images.size(3)))
            g_loss_l1 = l1_loss(generated_images, resized_real_images)
            g_loss_perceptual = perceptual_loss(generated_images, resized_real_images)
            
            g_loss = lambda_adv * g_loss_adv + lambda_l1 * g_loss_l1 + lambda_perceptual * g_loss_perceptual
            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            g_optimizer.step()
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            discriminator_output_real = discriminator(resized_real_images)
            discriminator_output_fake_detached = discriminator(generated_images.detach())

            real_loss = adversarial_loss(discriminator_output_real, valid)
            fake_loss = adversarial_loss(discriminator_output_fake_detached, fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            d_optimizer.step()
            
            progress_bar.set_description(
                f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                f"Generator Loss: {g_loss.item():.4f} Discriminator Loss: {d_loss.item():.4f}"
            )
            
            # Save and display some example generated images every few batches
            if i % display_interval == 0:
                save_and_display_images(epoch, i, input_images, resized_real_images, generated_images)

            # Debugging: Print loss values and gradient statistics occasionally
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}: g_loss_adv={g_loss_adv.item()}, g_loss_l1={g_loss_l1.item()}, g_loss_perceptual={g_loss_perceptual.item()}, d_loss={d_loss.item()}")
                
                # Print gradients and weight statistics
                for name, param in generator.named_parameters():
                    if param.grad is not None:
                        print(f"Generator {name} - grad: {param.grad.norm()}, weight: {param.data.norm()}")
                
                for name, param in discriminator.named_parameters():
                    if param.grad is not None:
                        print(f"Discriminator {name} - grad: {param.grad.norm()}, weight: {param.data.norm()}")

        # Save model checkpoints
        torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')

        # Step the learning rate scheduler
        g_scheduler.step()
        d_scheduler.step()

# Main script execution
if __name__ == '__main__':
    # Define the dataset and dataloader
    input_dir = 'coco_preprocessed/inputs_small'
    output_dir = 'coco_preprocessed/outputs_small'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CustomDataset(input_dir, output_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Initialize models and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    perceptual_loss = PerceptualLoss().to(device)

    # Train the model
    num_epochs = 100
    train_model(generator, discriminator, perceptual_loss, dataloader, num_epochs, device)
