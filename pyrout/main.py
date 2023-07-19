import numpy as np

def rsdr(residuals, n_params_fit=0):
    """
    Computes the Robust Standard Deviation of the Residuals (RSDR)
    
    By Eq. 1 of Motulsky and Brown's paper, RSDR = P68*N/(N-K), 
    where P68 percentile of the absolute value of the residuals, 
    N the number of data points, K the number of parameters fit by nonlinear regression. 

    Parameters
    ----------
    data : array-like
        The input data values.

    n_params_fit : float
        the number of parameters fit by nonlinear regression. 

    Returns
    -------
    float
        The computed Robust Standard Deviation of Residuals (RSDR). 

    """
    sorted_abs_residuals = np.sort(np.abs(np.array(residuals)))
    index = int(len(sorted_abs_residuals) * 0.6872)
    p68 = sorted_abs_residuals[index]
    n = len(residuals)
    return p68*n/(n - n_params_fit)

def lorentzian_merit(params, model_func, data_points):
    """
    Compute the Lorentzian merit given data points and the model function to fit.

    By Eq. 8 of Motulsky and Brown's paper, Lorentzian Merit = sum(ln(1 + (D/RSDR)^2)). 

    Parameters
    ----------
    params : array-like
        The parameters of the model_func.

    model_func : callable
        The model function to be fitted using nonlinear regression.

    data_points : array-like
        The data points to be used for fitting the model function.

    Returns
    -------
    float
        The computed Lorentzian merit.

    """
    # Perform nonlinear regression and compute residuals
    params = nonlinear_regression(model_func, data_points)
    residuals = compute_residuals(model_func, data_points, params)

    # Compute RSDR
    rsdr = rsdr(residuals, len(params))

    # Compute Lorentzian merit
    squared_residuals = np.square(residuals)
    merit = np.sum(np.log(1 + squared_residuals / (rsdr ** 2)))
    return merit


def svi_black_scholes_call(K, r, a, b, rho, m, sigma):
    # Compute the SVI-Black Scholes formula with spot price version
    # Spot price (S) and time to maturity (T) are hardcoded
    
    S = 4267.52                 # ^SPX close on 9/29/2023
    T = 0.31232876712328766     # time to maturity by (datetime.date(2023, 9, 29) - datetime.date(2023, 6, 7)).days/365.
    
    w = a + b * (rho * (np.log(K / S) - m) + np.sqrt((np.log(K / S) - m) ** 2 + sigma ** 2))
    d1 = (np.log(S / K) + (r + 0.5 * w) * T) / np.sqrt(w * T)
    d2 = d1 - np.sqrt(w * T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price


if __name__ == '__main__':
    a=1

    data = [(800.0,3470.3), (1000.0,3273.6), (1200.0,3077.1), (1400.0,2880.6), (1500.0,2782.1), (2000.0,2172.8), (2600.0,1496.6), (2925.0,1281.2), (2950.0,1215.5), (2975.0,1100.5), (3000.0,1314.1), (3025.0,1252.2), (3075.0,944.3), (3100.0,967.0), (3125.0,900.2), (3150.0,1168.5), (3175.0,867.2), (3200.0,1006.8), (3225.0,813.0), (3250.0,975.4), (3275.0,781.3), (3300.0,1023.3), (3325.0,871.1), (3350.0,718.3), (3375.0,951.6), (3380.0,946.5), (3390.0,936.9), (3400.0,927.6), (3420.0,817.4), (3425.0,656.9), (3450.0,832.4), (3475.0,607.3), (3480.0,851.3), (3490.0,841.8), (3500.0,832.3), (3510.0,789.0), (3525.0,808.6), (3530.0,803.6), (3550.0,784.7), (3575.0,761.1), (3600.0,737.6), (3610.0,728.4), (3625.0,714.4), (3630.0,709.5), (3640.0,585.4), (3650.0,690.8), (3655.0,637.3), (3675.0,667.7), (3690.0,542.5), (3700.0,643.9), (3705.0,529.8), (3725.0,621.5), (3750.0,597.9), (3760.0,589.3), (3770.0,580.1), (3775.0,575.6), (3780.0,571.1), (3800.0,552.3), (3805.0,514.5), (3825.0,478.4), (3830.0,525.7), (3835.0,490.7), (3845.0,400.1), (3850.0,507.2), (3855.0,503.2), (3860.0,430.7), (3870.0,462.7), (3875.0,485.3), (3880.0,367.7), (3895.0,437.4), (3900.0,463.1), (3910.0,373.8), (3915.0,449.7), (3920.0,420.0), (3925.0,440.6), (3930.0,412.0), (3935.0,432.1), (3940.0,427.5), (3945.0,423.3), (3950.0,418.9), (3955.0,433.2), (3960.0,410.2), (3965.0,406.0), (3970.0,401.4), (3975.0,397.1), (3980.0,393.0), (3985.0,388.6), (3990.0,384.6), (3995.0,380.0), (4000.0,376.0), (4005.0,390.3), (4010.0,367.5), (4015.0,381.9), (4020.0,359.0), (4025.0,354.6), (4030.0,350.3), (4035.0,346.4), (4040.0,342.1), (4045.0,338.0), (4050.0,333.5), (4055.0,307.0), (4060.0,325.1), (4065.0,320.9), (4070.0,316.8), (4075.0,312.7), (4080.0,308.6), (4085.0,304.9), (4090.0,300.5), (4095.0,296.3), (4100.0,292.3), (4105.0,288.1), (4110.0,284.2), (4115.0,280.1), (4120.0,276.1), (4125.0,272.1), (4130.0,268.2), (4135.0,264.1), (4140.0,260.2), (4145.0,256.3), (4150.0,252.4), (4155.0,248.4), (4160.0,243.2), (4165.0,239.1), (4170.0,235.3), (4175.0,231.5), (4180.0,227.7), (4185.0,223.9), (4190.0,220.0), (4195.0,216.5), (4200.0,212.5), (4205.0,208.9), (4210.0,205.2), (4215.0,201.7), (4220.0,197.9), (4225.0,194.4), (4230.0,190.8), (4235.0,187.2), (4240.0,183.5), (4245.0,180.0), (4250.0,176.4), (4255.0,173.1), (4260.0,169.7), (4265.0,166.1), (4270.0,162.8), (4275.0,159.3), (4280.0,156.0), (4285.0,152.8), (4290.0,149.2), (4295.0,146.1), (4300.0,142.7), (4305.0,139.6), (4315.0,133.2), (4320.0,130.1), (4325.0,127.2), (4330.0,124.0), (4345.0,115.3), (4350.0,112.4), (4355.0,109.7), (4370.0,101.2), (4375.0,98.5), (4390.0,90.7), (4395.0,88.2), (4400.0,85.7), (4405.0,83.3), (4410.0,80.9), (4415.0,78.5), (4420.0,76.2), (4425.0,74.0), (4430.0,71.6), (4440.0,67.5), (4445.0,65.4), (4450.0,63.3), (4460.0,59.4), (4470.0,55.7), (4475.0,53.9), (4500.0,45.4), (4525.0,38.3), (4550.0,31.9), (4575.0,26.5), (4590.0,23.7), (4600.0,21.7), (4650.0,14.7), (4700.0,9.7), (4750.0,6.4), (4800.0,4.1), (4900.0,1.85), (4950.0,1.35), (5000.0,1.0), (5100.0,0.6), (5200.0,0.4), (5300.0,0.25), (5400.0,0.2)]
    