import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
from model import Generator, PatchDiscriminator, PerceptualLoss
import matplotlib.pyplot as plt
from tqdm import tqdm

# Utility function to save and display images
def save_and_display_images(epoch, batch, inputs, expected_outputs, model_outputs, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)
    
    inputs = inputs.cpu().numpy().transpose((0, 2, 3, 1))
    expected_outputs = expected_outputs.cpu().numpy().transpose((0, 2, 3, 1))
    model_outputs = model_outputs.cpu().detach().numpy().transpose((0, 2, 3, 1))
    
    for i in range(min(len(inputs), 5)):  # Display up to 5 images
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow((inputs[i] * 0.5 + 0.5))
        ax[0].set_title("Input")
        ax[1].imshow((expected_outputs[i] * 0.5 + 0.5))
        ax[1].set_title("Expected Output")
        ax[2].imshow((model_outputs[i] * 0.5 + 0.5))
        ax[2].set_title("Model Output")
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_batch_{batch}_image_{i}.png"))
        plt.close()

# Function to train the model
def train_model(generator, discriminator, perceptual_loss, dataloader, num_epochs, device, display_interval=1):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))  # Lower LR for discriminator

    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    lambda_adv = 1.0
    lambda_l1 = .2
    lambda_perceptual = .2

    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in progress_bar:
            real_images = data['real'].to(device)
            input_images = data['input'].to(device)
            batch_size = real_images.size(0)
            
            # Generate target tensors
            valid = torch.ones((batch_size, 1, discriminator(real_images).size(2), discriminator(real_images).size(3)), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1, discriminator(real_images).size(2), discriminator(real_images).size(3)), requires_grad=False).to(device)
            
            # Train Generator
            g_optimizer.zero_grad()
            
            generated_images = generator(input_images)
            g_loss_adv = adversarial_loss(discriminator(generated_images), valid)
            
            # Resize real images to match the generated images for L1 loss
            resized_real_images = nn.functional.interpolate(real_images, size=(generated_images.size(2), generated_images.size(3)))
            g_loss_l1 = l1_loss(generated_images, resized_real_images)
            g_loss_perceptual = perceptual_loss(generated_images, resized_real_images)
            
            g_loss = lambda_adv * g_loss_adv + lambda_l1 * g_loss_l1 + lambda_perceptual * g_loss_perceptual
            g_loss.backward()
            g_optimizer.step()
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            real_loss = adversarial_loss(discriminator(resized_real_images), valid)
            fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            d_optimizer.step()
            
            progress_bar.set_description(
                f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                f"Generator Loss: {g_loss.item():.4f} Discriminator Loss: {d_loss.item():.4f}"
            )
            
            # Save and display some example generated images every 10 batches
            if i % display_interval == 0:  # Display frequently in the first epoch
                save_and_display_images(epoch, i, input_images, resized_real_images, generated_images)

            # Debugging: Print loss values occasionally
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}: g_loss_adv={g_loss_adv.item()}, g_loss_l1={g_loss_l1.item()}, g_loss_perceptual={g_loss_perceptual.item()}, d_loss={d_loss.item()}")

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
    generator = Generator().to(device)
    discriminator = PatchDiscriminator().to(device)
    perceptual_loss = PerceptualLoss().to(device)

    # Train the model
    num_epochs = 100
    train_model(generator, discriminator, perceptual_loss, dataloader, num_epochs, device)
