import numpy as np
import matplotlib.pyplot as plt

def generate_dummy_visuals():
    print("Generating high-resolution dummy plot for presentation...")
    
    # Simulate 10 seconds of time
    fs = 4000
    t = np.linspace(0, 10, 10 * fs, endpoint=False)

    # 1. Simulate Respiratory Signal (0.25 Hz -> ~15 Breaths per minute)
    # Smooth sine wave with some realistic sensor noise
    resp_signal = np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 0.08, len(t))

    # 2. Simulate Heart Signal PCG (1.2 Hz -> ~72 Beats per minute)  
    # Simulating the 'Lub-dub' (S1 and S2) heart sounds
    heart_signal = np.zeros_like(t) + np.random.normal(0, 0.05, len(t))
    
    for beat_t in np.arange(0.5, 10, 1/1.2):
        # S1 (Lub)
        idx1 = int(beat_t * fs)
        if idx1 + 100 < len(heart_signal):
            heart_signal[idx1:idx1+100] += np.hanning(100) * 1.5 
            
        # S2 (Dub)
        idx2 = int((beat_t + 0.3) * fs)
        if idx2 + 80 < len(heart_signal):
            heart_signal[idx2:idx2+80] += np.hanning(80) * 0.9 

    # Plotting using a clean, modern aesthetic for presentations
    plt.style.use('dark_background') # Looks great on slides
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Plot Heart Signal
    ax1.plot(t, heart_signal, color='cyan', alpha=0.8, linewidth=1.5)
    ax1.set_title("AI-Filtered Heart Signal (PCG Envelope) - Heart Rate: 72 BPM", fontsize=14, pad=10)
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.15)
    ax1.margins(x=0)

    # Plot Respiratory Signal
    ax2.plot(t, resp_signal, color='lime', alpha=0.8, linewidth=2)
    ax2.set_title("AI-Filtered Lung Signal - Respiratory Rate: 15 BPM", fontsize=14, pad=10)
    # Fill under the breathing curve for better visual effect
    ax2.fill_between(t, resp_signal, 0, where=(resp_signal > 0), color='lime', alpha=0.2)
    ax2.set_xlabel("Time (Seconds)", fontsize=12)
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.15)
    ax2.margins(x=0)

    plt.tight_layout()
    
    save_path = 'presentation_dummy_slide.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Image saved as '{save_path}'. You can paste this directly into your presentation.")

if __name__ == "__main__":
    generate_dummy_visuals()
