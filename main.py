import numpy as np
import matplotlib.pyplot as plt
import math
from odrpack import odr_fit

#0. setting up m convention
m = [0,1,2,-1,-2, 3, -3] #all the angles will be in this order
#use m[0:5] most of the time

#1. Read from file
def ReadFile(fileName: str):
    data = np.genfromtxt(fileName, delimiter=",", encoding="UTF-8-sig")
    data = np.transpose(data)
    data = [item[~np.isnan(item)] for item in data]
    return data

def AngleTransform(data: np.array):
    temp = [[], [], []]
    for i in range(3):
        degree = data[2*i+1]
        minute = data[2*i+2]
        degree_0 = degree[0]
        minute_0 = minute[0]
        angle = (degree_0 - degree) + (minute_0 - minute) / 60
        temp[i] = np.radians(angle)
    return temp
        
calibration = AngleTransform(ReadFile("Calibration.csv"))
darkblue = AngleTransform(ReadFile("DarkBlue.csv"))
lightblue = AngleTransform(ReadFile("LightBlue.csv"))
green = AngleTransform(ReadFile("Green.csv"))
red = AngleTransform(ReadFile("Red.csv"))

#print(darkblue)
#print(len(darkblue))
#First, we need to determine an experimental a and phi (little angular grating offset)

m_active = np.array([0,1,2,-1,-2]) #the relevant m's for the calibration
lambda_Na_D1 = 588.9950e-9 
lambda_Na_D2 = 589.5924e-9 


#for now, let's just do an average (maybe use the intensity approach xinyi mentioned??)
lambda_average = (lambda_Na_D1 + lambda_Na_D2)/2 #need to account for uncertainty somehow

#the x axis will be theta and the y axis will be m* lambda
def experimental_grating_offset(theta, beta): #beta = [a, phi]
    return beta[0]*(np.sin(theta - beta[1]) - np.sin(beta[1]))


calibration_data = np.array(calibration)

theta_data = np.mean(calibration_data, axis = 0) #array of 5 theta means, one for each m

#print(len(theta_data))

mlambda_data = np.array(lambda_average * m_active)

#print(len(mlambda_data))

a_guess = (1/300)*1e-3
phi_guess = 0

beta0 = [a_guess, phi_guess]

#uncertainty of the vernier was 0.25 arcminutes

theta_delta = (0.25/60)*(np.pi/180) #converting vernier uncertainty to radians 
mlambda_delta = 0.0001e-9 * np.abs(m_active) #  multiply abs(m) by the NIST database uncertainty (0.0001e-9)

weight_theta = 1/(theta_delta**2)
weight_mlambda = 1/(mlambda_delta**2)
solution_calibration = odr_fit(experimental_grating_offset, theta_data, mlambda_data, beta0)

beta_solution = solution_calibration.beta

#print(beta_solution[0])
#print(beta_solution[1])

a_predicted = beta_solution[0]
phi_predicted = beta_solution[1]

#Great! Now we have computed a and phi, from there it goes much faster

#ODR also computed uncertainties for a and phi

a_delta, phi_delta = solution_calibration.sd_beta


#2- the actual code for determining the gas

#recall the order for m's : 0 1 2 -1 -2 3 -3

# A. let's start with dark blue

wavelengths_db = [] #this array will contain 12 dark blue wavelengths (because we are excluding m = 0, as well as m = +-3)
db_deltas = [] #array of 12 uncertainties, one for each wavelength
m_sequence = np.array([0, 1, 2, -1, -2])


for trial in darkblue:
    # let's ignore the m = +- 3 data points
    trimmed_trial = trial[:5]
    
    for m, theta in zip(m_sequence, trimmed_trial):
        if m == 0:
            continue  # skipping m = 0 to not get division by zero
            
        # this is just the equation 4 but rearranged to isolate lambda!
        lambd = (a_predicted / m) * (np.sin(theta - phi_predicted) - np.sin(phi_predicted))

        #let's compute the uncertainty (partial derivatives method)
        #check lab notebook for the computed derivatives themselves

        # the first term, with respect to a

        a_term = lambd/a_predicted * a_delta

        #the second term, with respect to theta

        theta_term = a_predicted/m * np.cos(theta - phi_predicted) * theta_delta

        #the third term, with respect to phi

        phi_term = -a_predicted/m * (np.cos(theta - phi_predicted) + np.cos(phi_predicted)) * phi_delta

        # pythagoras

        lambd_delta = np.sqrt(a_term**2 + theta_term**2 + phi_term**2)

        
        wavelengths_db.append(lambd)
        db_deltas.append(lambd_delta)




wavelengths_db = np.array(wavelengths_db)
db_deltas = np.array(db_deltas)
db_mean = np.mean(wavelengths_db)
db_std = np.std(wavelengths_db)
db_sem = db_std/np.sqrt(12)#this is the second part of the TOTAL uncertainty

#We now have the SEM and the 12 uncertainties, let's do a RMS (root mean square) on all 12 of them

rms_db_delta = np.sqrt(1/12 * np.sum(db_deltas**2))

#print(rms_db_delta)

#then finally, combine both contributions in quadrature

total_db_delta = np.sqrt(rms_db_delta**2 + db_sem**2)

print(db_mean)
print(total_db_delta)

# B. light blue

wavelengths_lb = [] 
lb_deltas = [] 


for trial in lightblue:
    
    trimmed_trial = trial[:5]
    
    for m, theta in zip(m_sequence, trimmed_trial):
        if m == 0:
            continue 
            
        
        lambd = (a_predicted / m) * (np.sin(theta - phi_predicted) - np.sin(phi_predicted))

        a_term = lambd/a_predicted * a_delta

        theta_term = a_predicted/m * np.cos(theta - phi_predicted) * theta_delta

        phi_term = -a_predicted/m * (np.cos(theta - phi_predicted) + np.cos(phi_predicted)) * phi_delta

        lambd_delta = np.sqrt(a_term**2 + theta_term**2 + phi_term**2)

        wavelengths_lb.append(lambd)
        lb_deltas.append(lambd_delta)


wavelengths_lb = np.array(wavelengths_lb)
lb_deltas = np.array(lb_deltas)
lb_mean = np.mean(wavelengths_lb)
lb_std = np.std(wavelengths_lb)
lb_sem = lb_std/np.sqrt(12)#this is the second part of the TOTAL uncertainty

rms_lb_delta = np.sqrt(1/12 * np.sum(lb_deltas**2))

total_lb_delta = np.sqrt(rms_lb_delta**2 + lb_sem**2)

print(lb_mean)
print(total_lb_delta)

# C. Green


wavelengths_g = [] 
g_deltas = [] 

for trial in green:
    
    trimmed_trial = trial[:5]
    
    for m, theta in zip(m_sequence, trimmed_trial):
        if m == 0:
            continue 
            
        
        lambd = (a_predicted / m) * (np.sin(theta - phi_predicted) - np.sin(phi_predicted))

        a_term = lambd/a_predicted * a_delta

        theta_term = a_predicted/m * np.cos(theta - phi_predicted) * theta_delta

        phi_term = -a_predicted/m * (np.cos(theta - phi_predicted) + np.cos(phi_predicted)) * phi_delta

        lambd_delta = np.sqrt(a_term**2 + theta_term**2 + phi_term**2)

        wavelengths_g.append(lambd)
        g_deltas.append(lambd_delta)


wavelengths_g = np.array(wavelengths_g)
g_deltas = np.array(g_deltas)
g_mean = np.mean(wavelengths_g)
g_std = np.std(wavelengths_g)
g_sem = g_std/np.sqrt(12)

rms_g_delta = np.sqrt(1/12 * np.sum(g_deltas**2))

total_g_delta = np.sqrt(rms_g_delta**2 + g_sem**2)

print(g_mean)
print(total_g_delta)

#D. Red
wavelengths_r = [] 
r_deltas = [] 


for trial in red:
    
    trimmed_trial = trial[:5]
    
    for m, theta in zip(m_sequence, trimmed_trial):
        if m == 0:
            continue 
            
        
        lambd = (a_predicted / m) * (np.sin(theta - phi_predicted) - np.sin(phi_predicted))

        a_term = lambd/a_predicted * a_delta

        theta_term = a_predicted/m * np.cos(theta - phi_predicted) * theta_delta

        phi_term = -a_predicted/m * (np.cos(theta - phi_predicted) + np.cos(phi_predicted)) * phi_delta

        lambd_delta = np.sqrt(a_term**2 + theta_term**2 + phi_term**2)

        wavelengths_r.append(lambd)
        r_deltas.append(lambd_delta)


wavelengths_r = np.array(wavelengths_r)
r_deltas = np.array(r_deltas)
r_mean = np.mean(wavelengths_r)
r_std = np.std(wavelengths_r)
r_sem = r_std/np.sqrt(12)#this is the second part of the TOTAL uncertainty

rms_r_delta = np.sqrt(1/12 * np.sum(r_deltas**2))

total_r_delta = np.sqrt(rms_r_delta**2 + r_sem**2)

print(r_mean)
print(total_r_delta)

#After consulting the NIST ASD database, it seems like a near perfect match with an element called Cadmium (Cd I)

#let's compute residuals across all the wavelengths

nist_wavelengths = np.array([467.81493e-9, 479.99123e-9, 508.58217e-9, 643.84695e-9]) #theoretical

residuals_db = wavelengths_db - nist_wavelengths[0]
residuals_lb = wavelengths_lb - nist_wavelengths[1]
residuals_g = wavelengths_g - nist_wavelengths[2]
residuals_r = wavelengths_r - nist_wavelengths[3]


plots_data = [
    (residuals_db, db_deltas, 'darkblue'),
    (residuals_lb, lb_deltas, 'deepskyblue'),
    (residuals_g, g_deltas, 'forestgreen'),
    (residuals_r, r_deltas, 'crimson')
]

for res, yerr, color in plots_data:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200, constrained_layout=True)
    
    ax.errorbar(range(len(res)), res, yerr=yerr, fmt='o', color=color, capsize=3, capthick=1)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, label = 'NIST reference wavelength')
    
    
    ax.set_xlabel('Trial index', fontsize=10)
    ax.set_ylabel('Residuals [nm]', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=6)
    
    plt.show()







