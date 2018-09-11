"""
The following code inculdes functions and routines that can be used for such an
analysis.

Code Organzation
----------------
There are 5 main classes. The classes are;
1. Correlation_Function
2. Covariance_Matrix
3. Power_Spectrum
4. BAO_Measurements
5. BAO_fit
6. Polynomial

The classes are very natural as they have attributes and methods associated
to deal with them.

*This is core of the work I did during my research internship at the
International Centre for Radio Astronomy Research (ICRAR) from 09/07/2018 to
02/09/2018. The work was supervised by Cullan Howlett. He is a cool guy.*

"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def s_square(x, s): # Function that is used for s**2 * xi
    try:
        return s * x * s.T
    except:
        return s**2 * x

class Polynomial(object):
    """
    @class

    Callable polynomial functions for fitting procedures

    """
    def poly3(x, p0,p1,p2,p3):
        return p0*x**3+p1*x**2+p2*x**1+p3

    def poly5(x, p0,p1,p2,p3,p4,p5):
        return p0*x**5+p1*x**4+p2*x**3+p3*x**2+p4*x**1+p5

    def poly7(x, p0,p1,p2,p3,p4,p5,p6,p7):
        return p0*x**7+p1*x**6+p2*x**5+p3*x**4+p4*x**3+p5*x**2+p6*x**1+p7

    def poly9(x, p0,p1,p2,p3,p4,p5,p6,p7,p8,p9):
        return p0*x**9+p1*x**8+p2*x**7+p3*x**6+p4*x**5+p5*x**4+p6*x**3+\
                p7*x**2+p8*x**1+p9

class Correlation_Function(object):
    """
    @class
    Class for dealing with galaxy 2 point correlation functions

    @param
    ### CLASS ATTRIBUTES ###
        s:          array-like (M,)
                    Distance values in correlation function

        xi:         array-like (M,)
                    Correlation function values

        xi_0:       array-like (M,)
                    Correlation function values

        xi_2:       array-like (M,)
                    Correlation function values

        xi_4:       array-like (M,)
                    Correlation function values

        s2xi:       array-like (M,)
                    Correlation function values mutiplied by s**2.
                    Calculated by s * xi * s.T

        covariance: Covariance_Matrix object
                    Stores information related to the covariance in a
                    Covariance_Matrix instance. Look Covariance_Matrix
                    doc for more details

        s2covariance: Covariance_Matrix object
                    Stores information related to the covariance of a
                    correlation function mulitplied by s**2 using a
                    Covariance_Matrix instance. Look Covariance_Matrix
                    doc for more details

    @return
    ### CLASS METHODS ###
        set_covariance: Setter for covariance in Correlation_Function object

        from_file:  Reads the correlation function from file and generates an
                    instance of Correlation_Function

        from_mock:  Reads the correlation function from Mock_Number given the
                    location of the file and generates an instance of
                    Correlation_Function

        from_mock_bin: A similar function to Correlation_Function.from_mock
                    but allows passing bin_size for file names

        from_ps:    Gets a Power_Spectrum instance and integrates to find the
                    correction funtion. Returns a Correlation_Function object

        limit_range: Generates a new instance of Correlation_Function with
                    limited range

        plot_data:  Plots the correlation function data points and errors.
                    returns a matplotlib subplot object

        plot_bestfit: Plots the best fit for the correlation function data
                    points and errors. Should pass function to fit.
                    returns a matplotlib subplot object

        plot_measurements: Plots the necessary measurements done on the
                    correlation function. Returns a matplotlib subplot object

    """
    def __init__(self, Mock_Number, s, xi, xi_0=0, xi_2=0, xi_4=0, \
                covar = 1):

        if not isinstance(covar, (np.ndarray, np.generic)):
            covar = np.identity(len(s))

        self.mock_number = Mock_Number
        self.s = s
        self.xi = xi
        self.xi_0 = xi_0
        self.xi_2 = xi_2
        self.xi_4 = xi_4
        self.s2xi = s * xi * s.T
        # if covar!= None:
        covar2 = covar.copy()
        self.covariance = Covariance_Matrix(covar)
        self.s2covariance = Covariance_Matrix(covar2)
        self.s2covariance.covar_matrix = \
                                    s * self.s2covariance.covar_matrix * s.T
        self.s2covariance.variance = s**2 * self.s2covariance.variance
        self.s2covariance.sigma = s**2 * self.s2covariance.sigma

    def set_covariance(self, covar):
        """
        *Acts weird*
        Setter for covariance in Correlation_Function class object

        Pass  Covariance_Matrix instance or an array-like object

        EXAMPLE:
            cf.set_covariance(bao.Covariance_Matrix.from_file("filename.txt"))

        """

        if isinstance(covar,Covariance_Matrix):
            covar = covar.covar_matrix
            covar2 = covar.copy()

        if isinstance(covar, (np.ndarray, np.generic)):
            self.covariance = Covariance_Matrix(covar)
            self.s2covariance = Covariance_Matrix(covar2)

        self.s2covariance.covar_matrix = \
                            self.s * self.s2covariance.covar_matrix * self.s.T
        self.s2covariance.variance = self.s**2 * self.s2covariance.variance
        self.s2covariance.sigma = self.s**2 * self.s2covariance.sigma

    def from_file(filepath, Mock_Number=000):
        """
        Reads the correlation function from file and generates an instance of
        Correlation_Function

        @example
        EXAMPLE:
            bao.Covariance_Matrix.from_file("filename.txt")
        """
        tfile = open(filepath,"r")
        s, xi, xi_0, xi_2, xi_4  = np.loadtxt(tfile, unpack=True)
        return Correlation_Function(Mock_Number, s, xi, xi_0, xi_2, xi_4)

    def from_mock(Mock_Number, Mock_location):
        """
        Reads the correlation function from Mock_Number given the location of
        the file and generates an instance of Correlation_Function

        ### EXAMPLE ###
            cf = bao.Correlation_Function.from_mock(mocknum, "mock_location/")
        """
        #Mock_location = "Mock_xi/"

        filename = Mock_location + "Mock_taipan_year1_v1_R19" + \
                    ('{:03}'.format(Mock_Number)) + ".xi"
        tfile = open(filename,"r")
        s, xi, xi_0, xi_2, xi_4  = np.loadtxt(tfile, unpack=True)
        return Correlation_Function(Mock_Number, s, xi, xi_0, xi_2, xi_4)

        #Mock_taipan_year1_v1.xi

    def from_mock_bin(Mock_Number, Mock_location, bin_size=5):
        """
        Similar function to Correlation_Function.from_mock but allows passing
        bin_size for file names

        EXAMPLE:
            cf = bao.Correlation_Function.from_mock_bin(mocknum, "mock_loc",
                                                        bin_size = 5)
        """
        #Mock_location = "Mock_taipan_year1_v1.xi"

        filename = Mock_location + "Mock_taipan_year1_v1_R19" + \
                    ('{:03}'.format(Mock_Number)) + ".xi_" + \
                    ('{:01}'.format(bin_size))
        tfile = open(filename,"r")
        s, xi, xi_0, xi_2, xi_4  = np.loadtxt(tfile, unpack=True)
        return Correlation_Function(Mock_Number, s, xi, xi_0, xi_2, xi_4)

    def ps2cf_integrand(k,s,a, p_curve):
        from scipy import special
        return 1/(2*(np.pi)**2) * k**2  * p_curve(k) * \
                        special.spherical_jn(0,k*s) * np.exp(-k**2 * a**2)

    def from_ps(ps, s_range=[0,200], alpha=2):
        """
        gets a Power_Spectrum instance and integrates according to the equation

            xi(s) = 1/(2*(np.pi)**2) * k**2  * P(k) * special.spherical_jn(0,k*s)
                * np.exp(-k**2 * alpha**2)

        where P(k) is the power spectrum and alpha is used as a damping factor
        to prevent the integrand to blow up in high k. The function then
        spits out an instance of Correlation_Function

        """
        from scipy import interpolate
        from scipy import integrate

        if type(alpha) is int:
            alpha = [alpha]

        s = np.linspace(s_range[0],s_range[1],60)
        p_curve = interpolate.interp1d(ps.k, ps.p, kind="cubic", \
                                        fill_value='extrapolate')

        cf = np.zeros(len(alpha),order='F')
        err = np.zeros(len(alpha),order='F')

        for i in range(len(s)):
            cf_val=[]
            err_val=[]
            #print(s[i])
            for j in range(len(alpha)):
                #print((s[i],a_values[j]))
                cf_val_a,err_val_a = integrate.quad(Correlation_Function.ps2cf_integrand, \
                                a=0.0001, b=100, args=(s[i], alpha[j], p_curve))

                cf_val = np.append(cf_val,cf_val_a)
                err_val = np.append(err_val,cf_val_a)
            #print(cf_val)
            cf = np.vstack((cf,cf_val))
            err = np.vstack((err,err_val))

        cf=np.delete(cf, 0, 0)
        err=np.delete(err, 0, 0)

        xi=np.zeros(cf.shape[0],order='F')
        for i in range(cf.shape[1]):
            xi = np.vstack((xi,cf[:,i]))

        np.delete(xi, np.s_[:0],axis=1)
        xi = xi.T
        xi=xi[:,1:]
        xi = xi.reshape((xi.shape[0],))

        cf = Correlation_Function(Mock_Number = 000, \
                                    s = s, xi = xi)
        return cf


    #Modifiers
    def limit_range(cf_old, s_range=[0,None], pos_range=[0,None]):
        """
        Generates a new instance of Correlation_Function with limited range
        """
        min_pos=0
        max_pos=len(cf_old.s)-1

        if s_range!=[0,None]:
            while cf_old.s[min_pos]<=s_range[0]:
                min_pos+=1
            while cf_old.s[max_pos]>=s_range[1]:
                max_pos-=1
            max_pos+=1

        elif pos_range!=[0,None]:
            min_pos = pos_range[0]
            max_pos = pos_range[1]

        new_s = cf_old.s[min_pos:max_pos]
        new_xi = cf_old.xi[min_pos:max_pos]
        new_s2xi = cf_old.s2xi[min_pos:max_pos]

        try:
            new_xi_0 = cf_old.xi_0[min_pos:max_pos]
            new_xi_2 = cf_old.xi_2[min_pos:max_pos]
            new_xi_4 = cf_old.xi_4[min_pos:max_pos]
            new_covariance = cf_old.covariance.covar_matrix[min_pos:max_pos,\
                                                            min_pos:max_pos]
            new_cf = Correlation_Function(Mock_Number = cf_old.mock_number, \
                                        s = new_s, xi = new_xi, xi_0 = new_xi_0, \
                                        xi_2 = new_xi_2, xi_4 = new_xi_4, \
                                        covar = new_covariance)

        except:
            new_covariance = cf_old.covariance.covar_matrix[min_pos:max_pos,\
                                                            min_pos:max_pos]
            new_cf = Correlation_Function(Mock_Number = cf_old.mock_number, \
                                    s = new_s, xi = new_xi, \
                                    covar = new_covariance)

        # new_cf = Correlation_Function(Mock_Number = cf_old.mock_number, \
        #                             s = new_s, xi = new_xi, xi_0 = new_xi_0, \
        #                             xi_2 = new_xi_2, xi_4 = new_xi_4, \
        #                             covar = new_covariance)
        return new_cf

    def plot_data(ax, cf, s2 = True, ebar=True):
        """
        Plots the correlation function and outputs a subplot object
        """
        if s2 == True:
            if ebar==True:
                ax.errorbar(cf.s, cf.s2xi, yerr=cf.s2covariance.sigma, fmt='ko')
            else:
                ax.plot(cf.s, cf.s2xi, fmt='ko')
            ax.set(ylabel=r"$s^{2} \xi (s)$", xlabel=r"$s$")
        else:
            if ebar==True:
                ax.errorbar(cf.s, cf.xi, yerr=cf.covariance.sigma, fmt='ko')
            else:
                ax.plot(cf.s, cf.xi, fmt='ko')
            ax.set(ylabel=r"$ \xi (s)$", xlabel=r"$s$")
        #ax.errorbar(s, s2xi, yerr=ye, fmt='o', capthick=5)

        return ax

    def plot_bestfit(ax, cf, fit_function, s2 = True, **kwargs):
        """
        Plots the correlation function and outputs a subplot object
        """
        x = cf.s
        if s2 == True:
            y = cf.s2xi
            cm = cf.s2covariance.covar_matrix
        else:
            y = cf.xi
            cm = cf.covariance.covar_matrix

        fit_coeff, fit_covar = BAO_fit.function_fit(x, y, \
                                        fit_function, cm)

        c_bestfit=np.poly1d(fit_coeff)

        new_x = np.linspace(x.min(),x.max(),50)
        try:
            ax.plot(new_x, c_bestfit(new_x), 'k', label= "Best Fit", **kwargs)
        except:
            ax.plot(new_x, c_bestfit(new_x), 'k',label= "Best Fit")

        if s2 == True:
            ax.set(ylabel=r"$s^{2} \xi (s)$", xlabel=r"$s$")
        else:
            ax.set(ylabel=r"$\xi (s)$", xlabel=r"$s$")
        return ax

    def plot_measurements(ax, cf, s2=True):
        """
        Plots the measuremnts from the input correlation function and outputs
        a subplot object
        """
        cf_measurement = BAO_fit.getBAO(cf, s2 = s2)

        if cf_measurement.peak_point is not np.nan:
            x_peak = cf_measurement.peak_point
        else: x_peak = np.nan
        if cf_measurement.dip_point is not np.nan:
            x_dip = cf_measurement.dip_point
        else: x_dip= np.nan
        if cf_measurement.linear_point is not np.nan:
            x_lp = cf_measurement.linear_point
        else: x_lp= np.nan
        if cf_measurement.linear_point_error is not np.nan:
            e_lp = cf_measurement.linear_point_error
        else: e_lp = np.nan
        if cf_measurement.inflection_point is not np.nan:
            x_ip = cf_measurement.inflection_point
        else: x_ip= np.nan
        if cf_measurement.inflection_point_error is not np.nan:
            e_ip = cf_measurement.inflection_point_error
        else: e_ip= np.nan

        fit_coeff = cf_measurement.fit_coeff
        c_bestfit=np.poly1d(fit_coeff)

        ax.plot(x_peak,c_bestfit(x_peak),'bo', mew = 3, label='Peak')
        ax.plot(x_dip,c_bestfit(x_dip),'ro', mew = 3, label='Dip')
        ax.plot(x_lp,c_bestfit(x_lp),'o', label='Linear Point', mew = 7)
        ax.plot(x_ip,c_bestfit(x_ip),'x', label='Inflection Point', mew = 7)

        if s2 == True:
            ax.set(ylabel=r"$s^{2} \xi (s)$", xlabel=r"$s$")
        else:
            ax.set(ylabel=r"$\xi (s)$", xlabel=r"$s$")
        return ax

class Power_Spectrum(object):
    """
    @class
    Class for dealing with the galaxy power spectrum

    ### CLASS ATTRIBUTES ###
        k:          array-like (M,)
                    Wave number values of the power spectrum

        p:          array-like (M,)
                    Power spectrum values correeposding to k values

        k_log10:    array-like (M,)
                    Wave number values of the power spectrum in log scale

        p_log10:    array-like (M,)
                    Power spectrum values correeposding to k values in log scale

    ### CLASS METHODS ###
        from_file:  Reads the power spectrum from file and generates an
                    instance of Power_Spectrum

        plot_ps:    Plots the power spectrum and returns a matplotlib subplot
                    object

    """
    def __init__(self, wave_number, power):
        self.k = wave_number
        self.p = power
        self.k_log10 = np.log10(wave_number)
        self.p_log10 = np.log10(power)

    #Initialization
    def from_file(filepath):
        """
        Reads the power_spectrum from file and generates an instance of
        Power_Spectrum
        """
        tfile=open(filepath,"r") #"camb_TAIPAN_matterpower_linear.dat"
        k, p  = np.loadtxt(tfile, unpack=True)
        return Power_Spectrum(k, p)

    def plot(self):
        """
        Plots the power spectrum and outputs a subplot object
        """

class Covariance_Matrix(object):
    """
    @class
    Class for reading and dealing with covariance matrices of correlation
    function

    ### CLASS ATTRIBUTES ###
        covar_matrix: array-like (M,M)
                    Covariance matrix of a correlation function

        variance:   array-like (M,)
                    Values in the diagonal of the covariance matrix

        sigma:      array-like (M,)
                    Standard deviation associated to a xi value. Found by sqrt
                    of variance

    ### CLASS METHODS ###
        from_file:  Reads the covariance matrix from file and generates an
                    instance of Covariance_Matrix

        from_cf_sample: Calculates a sample covariance matrix from a sample
                    of correlation functions in the sample_location

    """
    def __init__(self, array, sample_correction = False):

        #Calculates the covariance matrix correction factor and appliess it
        if sample_correction == True:
            #correction of the covariance matrix for finite number of simulations
            nmocks = 1000.
            nb = 15.0
            fac = np.true_divide((nmocks-nb-2.0),(nmocks-1.0))
            #array = (1.0/fac)*array
            array = np.true_divide(array,fac)

        self.covar_matrix = array
        self.variance = array.diagonal()
        self.sigma = np.sqrt(array.diagonal())

    def from_file(filepath):
        """
        Reads the power_spectrum from file and generates an instance of
        Power_Spectrum
        """
        cfile = open(filepath,"r")
        c_matrix = np.loadtxt(cfile, unpack=True)
        return Covariance_Matrix(c_matrix)

    def from_cf_sample(sample_location):
        """
        Calculates a sample covariance matrix from a sample of correlation
        functions in the sample_location

        ### PARAMETERS ###
            sample_location: string
                        The location of the correlation files the sample
                        covariance matrix has to be estimated

        ### RETURNS ###
            sample covariance: bao.Covariance_Matrix instance

        """
        mypath = sample_location #'Mock_taipan_year1_v1.xi/1'
        from os import listdir
        from os.path import isfile, join
        filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        filenames.sort()

        cf0 = Correlation_Function.from_file(filepath=(mypath+filenames[0]))
        r_mock, xi_null = cf0.s, cf0.xi
        nmocks = len(filenames)
        # k_mock = [0]

        cfs = []
        mock_all = []#np.zeros((nmocks,len(r_mock)))
        for i in range(nmocks):
            cf = Correlation_Function.from_file(filepath=\
                                                        (mypath+filenames[i]))
            # mock_all[i,len(k_mock):] = cf.xi
            mock_all.append(cf.xi)

        mock_all = np.asarray(mock_all)
        mockave = np.sum(mock_all,axis=0)/nmocks
        mockcov = np.zeros((len(r_mock),len(r_mock)))
        for i in range(len(r_mock)):
            mockcov[i,0:] = (np.sum(mock_all[0:,i,None]*mock_all,axis=0) - \
                                    nmocks*mockave[i]*mockave[0:])/(nmocks-1.0)

        return Covariance_Matrix(mockcov)

    def save_to_file(self, filename, filepath = ""):
        """
        Saves the covariance matrix to file

        Can be a Covariance_Matrix instance or array-like
        """
        c_m = 0
        if isinstance(self, Covariance_Matrix):
            c_m = self.covar_matrix
        elif isinstance(self, (np.ndarray, np.generic)):
            c_m = self
        else:
            print("error type")
        np.savetxt((filepath + filename), c_m)

class BAO_Measurements(object):
    """
    @class
    Contains the BAO measurements made from the fitting procedures

    ### CLASS ATTRIBUTES ###
        mock_number:integer
                    Stores the mock number

        linear_point: float
                    Value of the linear point measurement from the best fit.
                    linear point is calculated by taking the midpoint of the
                    measured peak and dip (derivative = 0)

        linear_point_error: float
                    Error associated with the linear point measurement

        inflection_point: float
                    Value of the inflection poitn measurement done from the
                    best fit. Inflection point is calculated by taking the
                    second derivative to be zero

        inflection_point_error: float
                    Error associated with the inflection point measurement

        peak_point: float
                    Point where the peak is observed. Calculated by taking
                    the derivative and cheking if the second derivative is
                    negative

        dip_point:  float
                    Point where the dip is observed. Calculated by taking
                    the derivative and cheking if the second derivative is
                    positive

        fit_coeff:  array-like (M,)
                    The list of coefficients from the best fit function

        fit_covar:  array-like (M,M)
                    The covariance matrix associated with the best fit
                    coefficients

        flag:       Boolean
                    flag set to True if issue is found when measurements
                    are made

        multi_lp:   Boolean
                    True if more than one linear point is found

    """

    def __init__(self, Mock_Number, \
                Linear_Point = np.nan, Error_LP = np.nan, \
                Inflection_Point = np.nan, Error_IP = np.nan, \
                Peak_Point = np.nan, Dip_Point = np.nan, \
                Fit_Coeff=[], Fit_Covar=[], Fit_Range = [np.nan,np.nan], \
                flag = False, multi_lp = [np.nan]):

        self.mock_number = Mock_Number
        self.linear_point = Linear_Point
        self.linear_point_error = Error_LP
        self.inflection_point = Inflection_Point
        self.inflection_point_error = Error_IP
        self.peak_point = Peak_Point
        self.dip_point = Dip_Point
        self.fit_coeff = Fit_Coeff
        self.fit_covar = Fit_Covar
        self.fit_range = Fit_Range
        self.flag =  flag
        self.multiple_lp = multi_lp

    def __str__(self):
        return ("({}, {}, {}, {}, {}, {})".format(self.mock_number, \
                                                self.linear_point, \
                                                self.linear_point_error, \
                                                self.inflection_point, \
                                                self.inflection_point_error, \
                                                self.flag))


class BAO_fit(object):
    """
    @class
    Contains the methods for BAO analysis by fitting the galaxy correlation
    function.

    ### CLASS METHODS ###
        npoly_fit:   Fitting correlation function with nth order polynomial

        function_fit: Fitting correlation function with user defined callable
                    function

        lp_find:    Find the linear point and the inflection point from the
                    coefficients of the polynomial fit to the correlation
                    function.

        mc_realization: Runs Monte Carlo realization of the linear point
                    measurement given number of times to estimate the error
                    in the measurements.

        getBAO:     The procedure that takes in a correlation function and
                    calculates all the measurements. Returns an instance of
                    BAO_Measurements

    """


    def npoly_fit(x, y, n):
        """
        Input a Correlation_Function class object with polynomial degree n

        * Plan to write a simple polynomail fit using scipy.polyfit

        ### PARAMETERS ###
            x:          array-like (M,)
                        Values of abscissa

            y:          array-like (M,)
                        Values of the ordinate

            function:   positive integer
                        order of polynomial to fit

        ### RETURNS ###
            fit_coeff:  array-like
                        Set of coefficients obtained by polynomial fitting

        """
        #s,s2xi = cf.s, cf.s2xi
        #polyfit

    def function_fit(x, y, function, covar=None):
        """
        Input a callable function with a covariance matrix instance to fit
        using scipy.curve_fit

        # Needs checking as the covariance matrix returned from curve_fit
        does not seem to be robust. MCMC implementation planned

        ### PARAMETERS ###
            x:          array-like (M,)
                        Values of abscissa

            y:          array-like (M,)
                        Values of the ordinate

            function:   callable function
                        Callablefunction for fitting

            covar:      array-like (M,M) (optional)
                        Set covariance matrix if applicable

        ### RETURNS ###
            fit_coeff:  array-like
                        Set of coefficients obtained by fitting the specified
                        function

            fit_covar:  array_like
                        Covariance matrix retunred from curve_fit routine

        """
        from scipy.optimize import curve_fit
        if isinstance(covar,(np.ndarray, np.generic)):
            fit_coeff, fit_covar = curve_fit(function, x, y, sigma=covar,
                                                            absolute_sigma=True)
        else:
            fit_coeff, fit_covar = curve_fit(function, x, y, absolute_sigma=True)
        return fit_coeff, fit_covar

    def find_lp_old(fit_coeff, fit_range, ip_output = True, \
                dip_range_prior=[60,120], peak_range_prior=[70,130], \
                ip_range_prior=[70,110]):
        """
        ################
        deprecated
        ################

        Find the linear point and the inflection point from the coefficients
        of the polynomial fit to the correlation function.

        ### PARAMETERS ###
            fit_coeff:  array-like (M,)
                        Values of the coefficients of the polynomial fit ordered
                        from a_n to a_0 where,
                            p(x) = (a_n * x**n) + ... + (a_1 * x) + (a_0)

            fit_range:  [a,b] where a<b and a and b are positive integer values
                        Takes in the range of s values the analysis should be done.
                        Recommended to use the same range used for fitting.

            ip_output:  Boolean (optional)
                        If True, outputs the inflection point together with the
                        linear point.

            dip_range_prior: [a,b] where a<b and a and b are positive integers
                        Specifying a prior for dip position

            peak_range_prior: [a,b] where a<b and a and b are positive integers
                        Specifying a prior for dip position

            ip_range_prior: [a,b] where a<b and a and b are positive integers
                        Specifying a prior for dip position

        ### RETURNS ###
            x_lp:       positive integer
                        position of the linear point
            x_ip:       positive integer (only if ip_output = True)
                        position of the inflection point

        """
        x_lp = [np.nan]
        x_ip = [np.nan]
        x_peak = [np.nan]
        x_dip = [np.nan]

        dip_range_prior = [fit_range[0]+10, fit_range[1]-20]
        peak_range_prior = [fit_range[0]+20, fit_range[1]-10]

        c_bestfit=np.poly1d(fit_coeff)

        #Calculating the critical points
        crit = c_bestfit.deriv(1).roots
        crit = np.real(crit[np.isreal(crit)])

        #Check if there are real critical points
        if len(crit)!=0:
            #Find Minimum and Maximum points
            test = c_bestfit.deriv(2)(crit)

            x_dip = crit[test>0]
            x_peak = crit[test<0]
            y_dip = [0]
            y_peak = [0]

            #For Sanity
            x_dip = x_dip[x_dip>dip_range_prior[0]]
            x_dip = x_dip[x_dip<dip_range_prior[1]]
            x_peak = x_peak[x_peak>peak_range_prior[0]]
            x_peak = x_peak[x_peak<peak_range_prior[1]]

            if ((len(x_dip)!=0) and (len(x_peak)!=0) ):
                #Take the highest minimum
                y_dip = c_bestfit(x_dip)
                y_dip_max = max(y_dip)
                y_dip_max_pos = [i for i, j in enumerate(y_dip) if \
                                                            j == y_dip_max]
                if len(y_dip_max_pos)==1:
                    x_dip = x_dip[y_dip_max_pos]
                    y_dip = c_bestfit(x_dip)

                #Take the lowest maximum
                y_peak = c_bestfit(x_peak)
                y_peak_min = min(y_peak)
                y_peak_min_pos = [i for i, j in enumerate(y_peak) if \
                                                            j == y_peak_min]
                if len(y_peak_min_pos)==1:
                    x_peak = x_peak[y_peak_min_pos]
                    y_peak = c_bestfit(x_peak)

                if (((y_peak - y_dip) / (x_peak - x_dip))>0):
                    #Calculate the Linear Point
                    x_lp = (x_peak+x_dip)/2

                ###########################
                # x_lp = x_lp[x_lp>75]
                # x_lp = x_lp[x_lp<115]
                # #if len(x_lp)==0: x_lp = [np.nan]
                ##########################

        if ip_output==True:
            #Look for real Inflection Points
            derive2 = c_bestfit.deriv(2).r
            #derive2 = derive2[np.isreal(derive2)]
            x_ip = np.real(derive2[derive2>0])

            if len(x_lp)!=0:
                x_ip = x_ip[abs(x_lp[0]-x_ip)<7]
            else:
                x_ip = x_ip[x_ip>range_prior[0]]
                x_ip = x_ip[x_ip<range_prior[1]]

            if len(x_ip)!=1:
                if len(x_ip)!=0:
                    midpos =int((len(x_ip)-1)/2)
                    x_ip = [x_ip[midpos]]
                else:
                    x_ip = [np.nan]
        #############################

        #print(x_lp.type)
        #print(x_ip.type)
        # print(x_lp)
        # x_lp = x_lp[x_lp>75]
        # x_lp = x_lp[x_lp<115]
        # x_ip = x_ip[x_ip>75]
        # x_ip = x_ip[x_ip<115]


        ############################

        if ip_output==False:
            return x_lp[0]

        try:
            return x_lp[0], x_ip[0], x_peak[0], x_dip[0]
        except:
            return x_lp[0], x_ip[0], x_peak, x_dip #np.nan, np.nan

    ############################################################################

    def find_lp(fit_coeff, fit_range, ip_output = True, \
                dip_range_prior=[60,120], peak_range_prior=[70,130], \
                range_prior=[70,110]):
        """
        Find the linear point and the inflection point from the coefficients
        of the polynomial fit to the correlation function.

        ### PARAMETERS ###
            fit_coeff:  array-like (M,)
                        Values of the coefficients of the polynomial fit ordered
                        from a_n to a_0 where,
                            p(x) = (a_n * x**n) + ... + (a_1 * x) + (a_0)

            fit_range:  [a,b] where a<b and a and b are positive integer values
                        Takes in the range of s values the analysis should be
                        done. Recommended to use the same range used for fitting.

            ip_output:  Boolean (optional)
                        If True, outputs the inflection point together with the
                        linear point.

            dip_range_prior: [a,b] where a<b and a and b are positive integers
                        Specifying a prior for dip position

            peak_range_prior: [a,b] where a<b and a and b are positive integers
                        Specifying a prior for dip position

            ip_range_prior: [a,b] where a<b and a and b are positive integers
                        Specifying a prior for dip position

        ### RETURNS ###
            x_lp:       float
                        Value of s of the linear point

            x_ip:       float
                        Value of s of the inflection point

            x_peak:     float
                        Value of s of the peak in the best fit

            x_dip:      float
                        Value of s of the dip in the best fit

            flag:       Boolean
                        Flag set to true if a problem is found

            multi_lp:   Boolean
                        True if more than one linear point is found
        """
        x_lp = np.nan
        x_ip = np.nan
        x_peak = [np.nan]
        x_dip = [np.nan]
        multi_lp = [np.nan]
        flag = False

        dip_range_prior = [fit_range[0]+10, fit_range[1]-20]
        peak_range_prior = [fit_range[0]+20, fit_range[1]-10]

        c_bestfit=np.poly1d(fit_coeff)

        #Calculating the critical points
        x_crit = c_bestfit.deriv(1).r
        x_crit = np.real(x_crit[np.isreal(x_crit)])
        x_crit.sort()
        #Check if there are real critical points
        if len(x_crit)!=0:
            #Find Minimum and Maximum points
            test = c_bestfit.deriv(2)(x_crit)
            x_dip = x_crit[test>0]
            x_peak = x_crit[test<0]
            x_dip = x_dip[x_dip>(fit_range[0])]
            x_dip = x_dip[x_dip<(fit_range[1])]
            x_peak = x_peak[x_peak>(fit_range[0])]
            x_peak = x_peak[x_peak<(fit_range[1])]

            # #############################
            #Find linear points with positive gradient
            lp = []
            for i in range(len(x_crit)-1):
                lp.append((x_crit[i+1]+x_crit[i])/2)
            lp = np.asarray(lp)
            lp_test = c_bestfit.deriv(1)(lp)
            lp = np.real(lp[lp_test>0])

            # #############################

            lp = lp[lp>fit_range[0]]
            lp = lp[lp<fit_range[1]]

            # y_crit = c_bestfit(x_crit)
            # m_crit = np.zeros(len(x_crit)-1)
            # for i in range(len(x_crit)-1):
            #     if ((y_crit[i+1]-y_crit[i]) / (x_crit[i+1]-x_crit[i])) >= 0:
            #         m_crit[i] = 1
            #     else:
            #         m_crit[i] = -1
            #
            # lp = []
            # for i in range(len(x_crit)-1):
            #     if m_crit[i] > 0:
            #         lp.append((x_crit[i+1]+x_crit[i])/2)

            #######################################

            # print("lp = {}".format(lp))


            if len(lp)!=0:
                # print("len(lp) = {}".format(len(lp)))
                if len(lp)>1:
                    flag = True
                    multi_lp = lp
                lp2=lp.copy()
                for i in range(len(lp)): lp2[i]-=90
                lp2=[abs(number) for number in lp2]
                lp_diff_min = min(lp2)
                # print("lp_diff_min = {}".format(lp_diff_min))
                lp_diff_min_pos = [i for i, j in enumerate(lp2) if \
                                                        abs(j) == lp_diff_min]
                # print("lp_diff_min_pos = {}".format(lp_diff_min_pos))
                # print("lp_diff_min_val = {}".format(lp[lp_diff_min_pos[0]]))
                if lp[lp_diff_min_pos[0]]>(range_prior[0]) and \
                                        lp[lp_diff_min_pos[0]]<(range_prior[1]):
                        x_lp = lp[lp_diff_min_pos[0]]
                        # print("x_lp = {}".format(x_lp))

        if ip_output==True:
            #Look for real Inflection Points
            derive2 = c_bestfit.deriv(2).roots
            test2 = c_bestfit.deriv(1)(derive2)
            #derive2 = derive2[np.isreal(derive2)]

            # x_ip = np.real(derive2[derive2>0])
            x_ip = np.real(derive2[test2>0])

            # x_ip = np.real(derive2)

            x_ip = x_ip[x_ip>fit_range[0]]
            x_ip = x_ip[x_ip<fit_range[1]]

            #################################################
            # x_ip = x_ip[x_ip>range_prior[0]]
            # x_ip = x_ip[x_ip<range_prior[1]]
            #################################################

            # #Check if linear point and inflection point is close enough
            # if x_lp is not np.nan:
            #     x_ip = x_ip[abs(x_lp-x_ip)<8]

            if len(x_ip)!=0:

                ip=x_ip.copy()
                for i in range(len(ip)): ip[i]-= 90
                # if x_lp is not np.nan:
                #     for i in range(len(ip)): ip[i]-=x_lp
                # else:
                #     for i in range(len(ip)): ip[i]-= 90
                #     #print('hi')
                ip=[abs(number) for number in ip]
                ip_diff_min = min(ip)
                ip_diff_min_pos = [i for i, j in enumerate(ip) if \
                                                        abs(j) == ip_diff_min]
                if x_ip[ip_diff_min_pos[0]]>range_prior[0] and \
                                        x_ip[ip_diff_min_pos[0]]<range_prior[1]:
                        x_ip = x_ip[ip_diff_min_pos[0]]
                        # if isinstance((x_ip), list): x_ip = x_ip[0]
            else:
                x_ip = np.nan

        return x_lp, x_ip, x_peak, x_dip, flag, multi_lp

    ############################################################################

    def mc_realization(fit_coeff, fit_range, fit_covar, i_number = 1000, \
                        ip_output=True):
        """
        Runs Monte Carlo realization of the linear point measurement
        given number of times to estimate the error in the measurements.

        ### PARAMETERS ###
            fit_coeff:  array-like (M,)
                        Values of the coefficients of the polynomial fit ordered
                        from a_n to a_0 where,
                            p(x) = (a_n * x**n) + ... + (a_1 * x) + (a_0)

            fit_covar:  array-like (N x N)
                        Covariance matrix obtained from the fitting procedure

            ip_output:  Boolean (optional)
                        If True, outputs the inflection point together with the
                        linear point.

            i_number:   positive integer
                        Number of Monte Carlo realizations to run
        ### RETURNS ###
            lp_e:       float
                        One sigma error on the linear point

            ip_e:       float
                        If ip_output is True, one sigma error on the inflection
                        point
        """
        mcr_coeff = np.random.multivariate_normal(fit_coeff, fit_covar,i_number)

        flag = False

        mcr_x_lp = np.array([])
        mcr_x_ip = np.array([])
        for i in range(mcr_coeff.shape[0]):
            mcr_x_lp_val, mcr_x_ip_val, x_peak, x_dip, flag, multi_lp = \
                                        BAO_fit.find_lp(mcr_coeff[i], fit_range)
            mcr_x_lp = np.append(mcr_x_lp, mcr_x_lp_val)
            mcr_x_ip = np.append(mcr_x_ip, mcr_x_ip_val)

        mcr_x_lp = mcr_x_lp[mcr_x_lp>0]
        mcr_x_ip = mcr_x_ip[mcr_x_ip>0]

        if ip_output==False:
            lp_e = np.std(mcr_x_lp)
            return lp_e, np.nan

        # print(list(mcr_x_ip))

        lp_e = np.std(mcr_x_lp)
        ip_e = np.std(mcr_x_ip)
        lp_mean = np.mean(mcr_x_lp)
        ip_mean = np.mean(mcr_x_ip)

        return lp_e, ip_e, lp_mean, ip_mean

    def getBAO(cf, s_range=[0,None], pos_range=[0,None], \
                                        function=Polynomial.poly5, s2 = False):
        """
        Finds the BAO measurements and returns a BAO_Measurements instance
        """

        x_lp = np.nan
        x_ip = np.nan
        x_peak = np.nan
        x_dip = np.nan
        lp_e = np.nan
        ip_e = np.nan
        mc_lp_m, mc_ip_m = np.nan, np.nan

        if s_range!=[0,None]: #run limit_range if a fitting range is specified
            correlation_f = Correlation_Function.limit_range(cf,s_range=s_range)

        elif pos_range!=[0,None]: #run limit_range if a fitting range is specified
            correlation_f = Correlation_Function.limit_range(cf,pos_range=pos_range)

        else: correlation_f = cf

        fit_range =[correlation_f.s.min(),correlation_f.s.max()]

        x = correlation_f.s
        c_m = None
        if s2 == True: #whether to fit s**2 * xi or xi
            y = correlation_f.s2xi
            try:
                c_m = correlation_f.s2covariance.covar_matrix
            except:
                pass
        else:
            y = correlation_f.xi
            try:
                c_m = correlation_f.covariance.covar_matrix
            except:
                pass


        # Fitting procedure using function_fit
        fit_coeff, fit_covar = BAO_fit.function_fit(x, y, \
                                                function, c_m)

        #Make the measurements from the fit
        x_lp, x_ip, x_peak, x_dip, flag, multi_lp = BAO_fit.find_lp(fit_coeff, \
                                                    fit_range=fit_range)

        #Calculate the error on the measurements
        ######################################################################
        # Calculating the correction factor of the error due to finite sampling
        nmocks = 1000
        nb = len(correlation_f.s)
        nsamp = 6
        if function == Polynomial.poly5: nsamp = 6
        if function == Polynomial.poly7: nsamp = 8
        if function == Polynomial.poly9: nsamp = 10

        corrfacA = 1.0/((nmocks - nb - 1.0)*(nmocks - nb - 4.0))
        corrfacB = corrfacA*(nmocks - nb - 2.0)

        corrfac = (1.0 + corrfacB*(nb - nsamp))/ \
                    (1.0 + 2.0*corrfacA + corrfacB*(nsamp+1.0))

        mc_iteration = 1000

        # Calculate error if the values were measured
        if not(x_lp is np.nan) and (x_ip is np.nan):
            lp_e, ip_e, mc_lp_m, mc_ip_m = BAO_fit.mc_realization(\
                                            fit_coeff=fit_coeff, \
                                            fit_range=fit_range, \
                                            fit_covar=fit_covar, \
                                            i_number = mc_iteration)
            ip_e = np.nan
            lp_e *= np.sqrt(corrfac)

        if (x_lp is np.nan) and not(x_ip is np.nan):
            lp_e, ip_e, mc_lp_m, mc_ip_m  = BAO_fit.mc_realization(\
                                            fit_coeff=fit_coeff, \
                                            fit_range=fit_range, \
                                            fit_covar=fit_covar, \
                                            i_number = mc_iteration)
            lp_e = np.nan
            ip_e *= np.sqrt(corrfac)

        if not(x_lp is np.nan) and not(x_ip is np.nan):
            lp_e, ip_e, mc_lp_m, mc_ip_m  = BAO_fit.mc_realization(\
                                            fit_coeff=fit_coeff, \
                                            fit_range=fit_range, \
                                            fit_covar=fit_covar, \
                                            i_number = mc_iteration)

            # multiplying the error correction for finite sampling
            lp_e *= np.sqrt(corrfac)
            ip_e *= np.sqrt(corrfac)

        # Remove if mean of mc realization is more than 1 sigma off
        # if abs(x_lp - mc_lp_m) > lp_e:
        #     x_lp, lp_e = np.nan, np.nan
        #     flag = True

        # Remove measurements if the mean measuremnt from MC realization
        # is off by 1 sigma
        try:
            if abs(x_ip - mc_ip_m) > ip_e:
                x_ip, ip_e = np.nan, np.nan
                flag = True
        except:
            # print(x_ip)
            x_ip = x_ip[0]
            if abs(x_ip - mc_ip_m) > ip_e:
                x_ip, ip_e = np.nan, np.nan
                flag = True

        # Setting the measurements to a BAO_Measurements instance
        measurement = BAO_Measurements(correlation_f.mock_number, \
                                    Linear_Point = x_lp, Error_LP = lp_e, \
                                    Inflection_Point = x_ip, Error_IP = ip_e, \
                                    Peak_Point = x_peak, Dip_Point = x_dip, \
                                    Fit_Coeff = fit_coeff, Fit_Covar = fit_covar,\
                                    Fit_Range = fit_range, flag = flag, \
                                    multi_lp = multi_lp)
        return measurement
