import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class Bjontegaard_Delta:
    def compute_BD_PSNR(self, model1, model2):
        log_bitrates1 = np.log10(model1.bitrates)
        log_bitrates2 = np.log10(model2.bitrates)

        # Get bounds for integration
        rL = np.max([np.min(log_bitrates1), np.min(log_bitrates2)])
        rH = np.min([np.max(log_bitrates1), np.max(log_bitrates2)])

        # Integrals of the polynomial in log space
        P1 = np.poly1d(np.polyint(model1.parameters_PSNR))
        P2 = np.poly1d(np.polyint(model2.parameters_PSNR))

        bd_delta_PSNR = 1 / (rH-rL) * ( (P2(rH)-P1(rH)) - (P2(rL)-P1(rL)) )
        return bd_delta_PSNR

    def compute_BD_Rate(self, model1, model2):
        D1 = model1.psnr_values
        D2 = model2.psnr_values
        
        # Get bounds for integration
        DL = np.max([np.min(D1), np.min(D2)])
        DH = np.min([np.max(D1), np.max(D2)])

        # Integrals of the polynomial
        P1 = np.poly1d(np.polyint(model1.parameters_Rate))
        P2 = np.poly1d(np.polyint(model2.parameters_Rate))

        exponent = 1 / (DH-DL) * ( (P2(DH)-P1(DH)) - (P2(DL)-P1(DL)) )
        bd_delta_rate = 10**(exponent) - 1
        return bd_delta_rate




class Bjontegaard_Model:
    def __init__(self, bitrates, psnr_values):
        self.bitrates = bitrates
        self.psnr_values = psnr_values

        self.parameters_PSNR = [0, 0, 0, 0]
        self.parameters_Rate = [0, 0, 0, 0]
        self.__update_model()

    def __update_model(self):
        logR = np.log10(self.bitrates) 
        self.parameters_PSNR = np.polyfit(logR, self.psnr_values, 3)
        self.parameters_Rate = np.polyfit(self.psnr_values, logR, 3)

    def evaluate(self, R):
        logR = np.log10(R)
        p = np.poly1d(self.parameters_PSNR)
        value = p(logR)
        return value

    def evaluate_rate(self, R):
        # This seems not correct
        p = np.poly1d(self.parameters_Rate)
        value = p(R)
        #value = 10**value
        return value

    def plot(self, ax):
        xdata = np.linspace(np.min(self.bitrates), np.max(self.bitrates), 100)
        p = np.poly1d(self.parameters_PSNR)

        ax.scatter(self.bitrates, self.psnr_values)
        ax.plot(xdata, p(np.log10(xdata)))

    def get_plot_data(self):
        xdata = np.linspace(np.min(self.bitrates), np.max(self.bitrates), 100)
        p = np.poly1d(self.parameters_PSNR)
        ydata = p(np.log10(xdata))
        return self.bitrates, self.psnr_values, xdata, ydata


if __name__ == "__main__":
    # Results of Zhu et al. ," View-Dependent Dynamic Point Cloud Compression"
    # on soldier, D1
    bitrates1 = [22.35, 12.93, 8.27, 4.53]
    bitrates2 = [24.35, 13.93, 9.27, 6.53]
    d1 = [71.17, 69.54, 67.62, 65.77]
    metric1 = Bjontegaard_Model(bitrates1, d1)
    metric2 = Bjontegaard_Model(bitrates2, d1)
    fig, ax = plt.subplots()
    metric1.plot(ax)
    metric2.plot(ax)
    BD_Delta = Bjontegaard_Delta()
    BD_Delta.compute_BD_PSNR(metric1, metric2)
    BD_Delta.compute_BD_Rate(metric1, metric2)
    BD_Delta.compute_BD_PSNR(metric2, metric1)
    BD_Delta.compute_BD_Rate(metric2, metric1)
    plt.show()
