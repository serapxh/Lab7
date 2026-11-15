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
print(calibration)
#print(darkblue)
#print(lightblue)
#print(green)
#print(red)

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

x_data = np.mean(calibration_data, axis = 0) #array of 5 theta means, one for each m

print(len(x_data))

y_data = np.array(lambda_average * m_active)

print(len(y_data))

a_guess = (1/300)*1e-3
phi_guess = 0

beta0 = [a_guess, phi_guess]
solution_calibration = odr_fit(experimental_grating_offset, x_data, y_data, beta0)


beta_solution = solution_calibration.beta

print(beta_solution[0])
print(beta_solution[1])

a_predicted = beta_solution[0]
phi_predicted = beta_solution[1]

#Great! Now we have computed a and phi, from there it goes much faster






