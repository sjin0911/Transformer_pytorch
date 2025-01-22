from torch.utils.tensorboard import SummaryWriter

Writer = SummaryWriter(log_dir="./runs/test")
print("Writer initialized")

for i in range(10):
    print(f"Writing scalar: step {i}, value {i*0.1}")
    Writer.add_scalar("Loss/test", i * 0.1, i)

Writer.close()
print("Writer closed")
