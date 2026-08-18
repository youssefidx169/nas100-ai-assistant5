
import matplotlib.pyplot as plt

def plot_support_resistance_zones(zones):
    fig, ax = plt.subplots(figsize=(10, 6))
    for z in zones:
        ax.axhspan(z['support'], z['resistance'], alpha=0.3)
    ax.set_title("Support and Resistance Zones")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    return fig
