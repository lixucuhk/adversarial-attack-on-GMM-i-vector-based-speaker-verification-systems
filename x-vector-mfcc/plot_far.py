import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('Agg')

plt.figure()  # an empty figure with no axes
# fig.suptitle('No axes on this figure')  # Add a title so we know which it is

x = np.array([0.0, 0.3, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0])
y_f = np.array([7.20, 8.97, 14.24, 47.75, 64.30, 67.96, 62.55, 31.41])
y_m = np.array([6.62, 8.46, 13.89, 52.16, 57.20, 29.57, 21.63, 19.33])
y_b = np.array([6.62, 7.34, 9.45, 23.92, 33.56, 37.48, 36.78, 30.80])

plt.plot(x, y_f, 'kp-', label='LPMS-ivec attacks MFCC-ivec')
plt.plot(x, y_m, 'cx--', label='MFCC-ivec attacks MFCC-xvec')
plt.plot(x, y_b, 'mo:', label='LPMS-ivec attacks MFCC-xvec')

# plt.plot(x, y_f, 'kp-', label='1')
# plt.plot(x, y_m, 'cx--', label='2')
# plt.plot(x, y_b, 'mo:', label='3')

plt.xlabel(r'perturbation degree $\epsilon$')
plt.ylabel('FAR (%)')

# plt.title("False acceptance rate (%) of the target models under different perturbation degree. \
# 	       The operation point is fixed at the EER point of the original system. ")

plt.legend()

plt.savefig('./black_box_far.png')
