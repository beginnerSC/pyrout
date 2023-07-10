def rsdr(residuals, n_params_fit=0):
    """
    Robust Standard Deviation of the Residuals (RSDR)
    
    By Eq. 1 of Motulsky and Brown's paper, RSDR = P68*N/(N-K), where P68 percentile of the absolute value of the residuals, N the number of data points, K the number of parameters fit by nonlinear regression. 

    Parameters
    ----------
    data : array-like
        The input data values.

    n_params_fit : float
        the number of parameters fit by nonlinear regression. 

    Returns
    -------
    float
        Robust Standard Deviation of Residuals. 

    """
    sorted_data = np.sort(np.array(residuals).abs())
    index = int(len(sorted_data) * 0.6872)
    percentile_68 = sorted_data[index]
    n = len(residuals)
    return percentile_68*n/(n - n_params_fit)

if __name__ == '__main__':
    a=1