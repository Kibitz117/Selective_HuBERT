import numpy as np

# Generate audio samples
num_samples = 100
duration = 1  # in seconds
sample_rate = 44100

# Generate samples from two audio distributions
queen_samples = np.random.normal(0.5, 0.2, (num_samples, duration * sample_rate))
no_queen_samples = np.random.normal(0.0, 0.1, (num_samples, duration * sample_rate))

# Generate labels
queen_labels = np.ones(num_samples)  # 1 for queen samples
no_queen_labels = np.zeros(num_samples)  # 0 for no queen samples

# Mix the samples and labels
samples = np.concatenate([queen_samples, no_queen_samples], axis=0)
labels = np.concatenate([queen_labels, no_queen_labels], axis=0)

# Normalize the samples to the range expected by HuBERT (-1.0 to 1.0)
normalized_samples = samples / np.max(np.abs(samples))

# Save the waveforms and the labels
np.save("data/queen_and_no_queen_waveforms.npy", normalized_samples)
np.save("data/queen_and_no_queen_labels.npy", labels)


# # Plot the audio waveforms
# plt.figure(figsize=(12, 4))
# plt.plot(waveforms[0:256], label="Queen hive")
# plt.plot(waveforms[100:128], label="No queen hive")
# plt.xlabel("Time (samples)")
# plt.ylabel("Amplitude")
# plt.title("Audio waveforms")
# plt.legend()
# plt.show()
