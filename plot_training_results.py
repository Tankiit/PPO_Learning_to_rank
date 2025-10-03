import matplotlib.pyplot as plt
import numpy as np
import os

# Training results from 10 epochs
epochs = list(range(1, 11))
train_loss = [0.0413, 0.0012, 0.0006, 0.0004, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001]
val_ndcg1 = [0.8655, 0.8782, 0.8800, 0.8800, 0.8818, 0.8818, 0.8818, 0.8818, 0.8818, 0.8836]
val_ndcg3 = [0.9180, 0.9238, 0.9243, 0.9242, 0.9243, 0.9243, 0.9242, 0.9242, 0.9242, 0.9242]
val_ndcg5 = [0.9398, 0.9437, 0.9426, 0.9423, 0.9425, 0.9424, 0.9425, 0.9424, 0.9424, 0.9424]
val_map = [0.8728, 0.8850, 0.8857, 0.8859, 0.8863, 0.8864, 0.8865, 0.8865, 0.8866, 0.8867]
val_spearman = [0.8583, 0.8654, 0.8632, 0.8626, 0.8629, 0.8628, 0.8630, 0.8629, 0.8628, 0.8629]

# Create output directory
os.makedirs('plots', exist_ok=True)

# Plot 1: Training Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss (MSE)', fontsize=12)
plt.title('Training Loss over 10 Epochs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('plots/training_loss.png', dpi=300, bbox_inches='tight')
print("✅ Saved: plots/training_loss.png")

# Plot 2: NDCG Metrics
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_ndcg1, 'r-o', linewidth=2, markersize=8, label='NDCG@1')
plt.plot(epochs, val_ndcg3, 'g-s', linewidth=2, markersize=8, label='NDCG@3')
plt.plot(epochs, val_ndcg5, 'b-^', linewidth=2, markersize=8, label='NDCG@5')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('NDCG Score', fontsize=12)
plt.title('Validation NDCG Metrics over 10 Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.ylim(0.85, 0.95)
plt.tight_layout()
plt.savefig('plots/ndcg_metrics.png', dpi=300, bbox_inches='tight')
print("✅ Saved: plots/ndcg_metrics.png")

# Plot 3: MAP and Spearman
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('MAP', color=color1, fontsize=12)
ax1.plot(epochs, val_map, 'b-o', linewidth=2, markersize=8, label='MAP')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Spearman Correlation', color=color2, fontsize=12)
ax2.plot(epochs, val_spearman, 'r-s', linewidth=2, markersize=8, label='Spearman')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Validation MAP and Spearman Correlation over 10 Epochs', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('plots/map_spearman.png', dpi=300, bbox_inches='tight')
print("✅ Saved: plots/map_spearman.png")

# Plot 4: All Metrics Combined
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, 'b-o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(2, 2, 2)
plt.plot(epochs, val_ndcg5, 'g-o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('NDCG@5')
plt.title('Validation NDCG@5', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0.93, 0.945)

plt.subplot(2, 2, 3)
plt.plot(epochs, val_map, 'r-o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MAP')
plt.title('Validation MAP', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0.87, 0.89)

plt.subplot(2, 2, 4)
plt.plot(epochs, val_spearman, 'm-o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Spearman ρ')
plt.title('Validation Spearman Correlation', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0.855, 0.870)

plt.suptitle('Ranking Reward Model Training Summary (10 Epochs)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/all_metrics_combined.png', dpi=300, bbox_inches='tight')
print("✅ Saved: plots/all_metrics_combined.png")

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Best Epoch: 2")
print(f"Best NDCG@5: 0.9437")
print(f"Final Train Loss: {train_loss[-1]:.6f}")
print(f"Final Val NDCG@5: {val_ndcg5[-1]:.4f}")
print(f"Final Val MAP: {val_map[-1]:.4f}")
print(f"Final Val Spearman: {val_spearman[-1]:.4f}")
print("="*60)
print("\n✅ All plots saved to 'plots/' directory")
