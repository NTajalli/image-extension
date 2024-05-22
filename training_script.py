import torch, torchvision
from dataset import CustomDataset
from model import *

def train_model(generator, discriminator, perceptual_loss, dataloader, num_epochs, device, display_interval=10):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            real_images = data['real'].to(device)
            input_images = data['input'].to(device)
            batch_size = real_images.size(0)
            
            valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)
            
            # Train Generator
            g_optimizer.zero_grad()
            
            generated_images = generator(input_images)
            g_loss_adv = adversarial_loss(discriminator(generated_images), valid)
            g_loss_l1 = l1_loss(generated_images, real_images)
            g_loss_perceptual = perceptual_loss(generated_images, real_images)
            
            g_loss = g_loss_adv + g_loss_l1 + g_loss_perceptual
            g_loss.backward()
            g_optimizer.step()
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            real_loss = adversarial_loss(discriminator(real_images), valid)
            fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            d_optimizer.step()
            
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Generator Loss: {g_loss.item()} Discriminator Loss: {d_loss.item()}")
            
            # Save and display some example generated images periodically
            if i % display_interval == 0:
                save_and_display_images(epoch, i, input_images, real_images, generated_images)


# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
input_dir = 'coco_preprocessed/inputs'
output_dir = 'coco_preprocessed/outputs'
dataset = CustomDataset(input_dir, output_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Initialize models and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = PatchDiscriminator().to(device)
perceptual_loss = PerceptualLoss().to(device)

# Train the model
num_epochs = 100
train_model(generator, discriminator, perceptual_loss, dataloader, num_epochs, device)
