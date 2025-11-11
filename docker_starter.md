# Docker Desktop Setup Guide

## Step 1: Download Docker Desktop

1. Go to [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Click the **Download for Windows** button (or Mac/Linux depending on your system)
3. Wait for the installer to download (it's about 500MB)

## Step 2: Install Docker Desktop

1. Run the installer you just downloaded
2. Follow the installation wizard:
   - Check "Use WSL 2 instead of Hyper-V" (Windows users - recommended)
   - Accept the license agreement
   - Click "Install"
3. The installation may take several minutes
4. When prompted, restart your computer

## Step 3: Create a Docker Account

1. Go to [https://hub.docker.com/signup](https://hub.docker.com/signup)
2. Create a free Docker account:
   - Enter your email address
   - Choose a username
   - Create a password
3. Verify your email address by clicking the link sent to your inbox

## Step 4: Start Docker Desktop

1. Launch Docker Desktop from your applications
2. Sign in with your Docker account credentials
3. Accept the Docker Subscription Service Agreement
4. Wait for Docker Desktop to start (you'll see a whale icon in your system tray)
5. Once running, the whale icon will stop animating

## Step 5: Verify Docker Installation

1. Open a terminal (Command Prompt, PowerShell, or Terminal)
2. Run the following commands to verify Docker is working:

```bash
docker --version
```

You should see something like: `Docker version 24.x.x`

```bash
docker run hello-world
```

This will download and run a test container. You should see "Hello from Docker!" message.

## Step 6: Install Docker Extension for VS Code

1. Open Visual Studio Code
2. Click the Extensions icon in the left sidebar (or press `Ctrl+Shift+X` / `Cmd+Shift+X`)
3. Search for "Docker"
4. Find the extension by Microsoft (it should be the first result)
5. Click **Install**

## Step 7: Verify Docker Extension in VS Code

1. You should now see a Docker icon in the left sidebar of VS Code
2. Click the Docker icon to open the Docker view
3. You should see sections for:
   - Containers
   - Images
   - Registries
   - Networks
   - Volumes

## Common Issues and Solutions

### Windows Users: WSL 2 Not Installed

If you see an error about WSL 2:
1. Open PowerShell as Administrator
2. Run: `wsl --install`
3. Restart your computer
4. Start Docker Desktop again

### Docker Desktop Won't Start

1. Make sure virtualization is enabled in your BIOS
2. Restart Docker Desktop
3. Check the Docker Desktop logs (Settings > Troubleshoot > View logs)

### Permission Denied Errors (Linux/Mac)

Run: `sudo usermod -aG docker $USER`

Then log out and log back in.

## You're Ready!

Once all steps are complete, you're ready to work with Docker containers in this course!
