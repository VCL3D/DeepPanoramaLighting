import math
import numpy as np
import time
import torch
import torchvision

#Convolve the SH coefficients with a low pass filter in spatial domain
def deringing(coeffs, window):
    deringed_coeffs = torch.zeros_like(coeffs)
    deringed_coeffs[:, 0] += coeffs[:, 0]
    deringed_coeffs[:, 1:1 + 3] += \
        coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
    deringed_coeffs[:, 4:4 + 5] += \
        coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
    return deringed_coeffs

# Spherical harmonics functions
def P(l, m, x):
	pmm = 1.0
	if(m>0):
		somx2 = np.sqrt((1.0-x)*(1.0+x))
		fact = 1.0
		for i in range(1,m+1):
			pmm *= (-fact) * somx2
			fact += 2.0
	
	if(l==m):
		return pmm * np.ones(x.shape)
	
	pmmp1 = x * (2.0*m+1.0) * pmm
	
	if(l==m+1):
		return pmmp1
	
	pll = np.zeros(x.shape)
	for ll in range(m+2, l+1):
		pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m)
		pmm = pmmp1
		pmmp1 = pll
	
	return pll

def factorial(x):
	if(x == 0):
		return 1.0
	return x * factorial(x-1)

def K(l, m):
	return np.sqrt( ((2 * l + 1) * factorial(l-m)) / (4*np.pi*factorial(l+m)) )

def SH(l, m, theta, phi):
	sqrt2 = np.sqrt(2.0)
	if(m==0):
		if np.isscalar(phi):
			return K(l,m)*P(l,m,np.cos(theta))
		else:
			return K(l,m)*P(l,m,np.cos(theta))*np.ones(phi.shape)
	elif(m>0):
		return sqrt2*K(l,m)*np.cos(m*phi)*P(l,m,np.cos(theta))
	else:
		return sqrt2*K(l,-m)*np.sin(-m*phi)*P(l,-m,np.cos(theta))

def shEvaluate(theta, phi, lmax):
	if np.isscalar(theta):
		coeffsMatrix = np.zeros((1,1,shTerms(lmax)))
	else:
		coeffsMatrix = np.zeros((theta.shape[0],phi.shape[0],shTerms(lmax)))

	for l in range(0,lmax+1):
		for m in range(-l,l+1):
			index = shIndex(l, m)
			coeffsMatrix[:,:,index] = SH(l, m, theta, phi)
	return coeffsMatrix

def getCoeeficientsMatrix(xres,lmax=2):
	yres = int(xres/2)
	# setup fast vectorisation
	x = np.arange(0,xres)
	y = np.arange(0,yres).reshape(yres,1)

	# Setup polar coordinates
	latLon = xy2ll(x,y,xres,yres)

	# Compute spherical harmonics. Apply thetaOffset due to EXR spherical coordiantes
	Ylm = shEvaluate(latLon[0], latLon[1], lmax)
	return Ylm

def shReconstructSignal(coeffs, sh_basis_matrix=None, width=512):
    if sh_basis_matrix is None:
        lmax = sh_lmax_from_terms(coeffs.shape[0])
        sh_basis_matrix = getCoeeficientsMatrix(width,lmax)
        sh_basis_matrix_t = torch.from_numpy(sh_basis_matrix).to(coeffs).float()
    return (torch.matmul(sh_basis_matrix_t,coeffs)).to(coeffs).float()

def shTerms(lmax):
	return (lmax + 1) * (lmax + 1)

def sh_lmax_from_terms(terms):
	return int(np.sqrt(terms)-1)

def shIndex(l, m):
	return l*l+l+m

def xy2ll(x,y,width,height):
	def yLocToLat(yLoc, height):
		return (yLoc / (float(height)/np.pi))
	def xLocToLon(xLoc, width):
		return (xLoc / (float(width)/(np.pi * 2)))
	return np.asarray([yLocToLat(y, height), xLocToLon(x, width)])

