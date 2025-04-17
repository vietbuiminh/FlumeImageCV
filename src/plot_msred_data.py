import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_msred_data(msred_data, title):
    """
    Plot the msred data with the specified title.

    Parameters:
    msred_data (pd.DataFrame): The msred data to plot.
    title (str): The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(msred_data['time'], msred_data['msred'], marker='o', linestyle='-', color='b')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MSRED')
    plt.grid()
    plt.show()

df_msred = pd.read_csv("data/msred_13b 17a_ABBA060115c.csv")
# This is SI-SII run from Kim & Muto and total of 7200s for each run
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JF000561
total_seconds = 7200
x = np.arange(0, total_seconds, total_seconds/len(df_msred))
Qs = 0.50 #cm^3/s
V = Qs * x # linear volume calculation
L1 = df_msred['L1'].astype(float)
L2 = df_msred['L2'].astype(float)
L3 = df_msred['L3'].astype(float)
angle = df_msred['angle'].astype(float)

# fig, axs = plt.subplots(5, 1, figsize=(10, 6), sharex=True)

# # Plot V
# axs[0].plot(x, V, color='r', label='V')
# # axs[0].set_ylabel(f'Expected Volume (cm³) Qs={Qs} cm³/s')
# axs[0].legend(loc='upper right')
# axs[0].grid()

# # Plot L1
# axs[1].plot(x, L1, color='g', label='L1')
# axs[1].set_ylabel('L1 (cm)')
# axs[1].legend(loc='upper right')
# axs[1].grid()

# # Plot L2
# axs[2].plot(x, L2, color='b', label='L2')
# axs[2].set_ylabel('L2 (cm)')
# axs[2].legend(loc='upper right')
# axs[2].grid()

# # Plot L3
# axs[3].plot(x, L3, color='m', label='L3')
# axs[3].set_ylabel('L3 (cm)')
# axs[3].legend(loc='upper right')
# axs[3].grid()

# # Plot angle
# axs[4].plot(x, angle, color='c', label='Angle')
# axs[4].set_xlabel('Time (s)')
# axs[4].set_ylabel('Angle (degrees)')
# axs[4].legend(loc='upper right')
# axs[4].grid()

# fig.subplots_adjust(top=2)  # Adjust the top to make space for the title
# fig.suptitle(f'Fixed SL 13b 17a_ABBA060115c, Qs={Qs} cm³/s', fontsize=16)
# plt.tight_layout()
# plt.show()

# Volume geometric model
# The sediment and water supplies were individually controlled by an sandglass-like funnel (potential error ≤ 2%) and a tube connected to multiple weirs (potential error ≤ 0.5%), respectively
long_slope = 0.22
L_mean = (L2+L3)/2
beta = long_slope
h = L_mean * beta * 1.0/np.sqrt(1.0+np.square(beta))
# r = L_mean * 1.0/np.sqrt(1.0+np.square(beta)) # this is the same with r down there
r = np.sqrt(np.square(L_mean) - np.square(h))

theta = np.radians(angle)
theta_star = 2.0 * np.arcsin(L_mean/r * np.sin(theta/2))

H = beta * L1/np.sqrt(1.0+np.square(beta)) - h
# R = L1/np.sqrt(1.0+np.square(beta)) # this is the same with R down there
R = np.sqrt(np.square(L1) - np.square(H+h))

# V_bottom = 0
# R_range = np.arange()

# cm^3
V_top = 1/3 * h * np.square(r) * (theta_star/2 - np.cos(theta_star/2)*np.sin(theta_star/2))
V_bottom = -H * theta_star * 1/2 * (np.square(r) - np.square(R))

# Plot it with the same x-axis and the V
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, V_top + V_bottom, color='orange', label='V extracted (top + bottom)')
ax.plot(x, V_top, color='green', label='V top')
ax.plot(x, V_bottom, color='blue', label='V bottom')
ax.plot(x, V, color='r', label='V expected')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Volume (cm³)')
ax.set_title(f'Volume geometric model, Qs={Qs} cm³/s')
ax.legend(loc='upper left')
ax.grid()
plt.tight_layout()
plt.show()
