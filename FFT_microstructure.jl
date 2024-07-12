# Necessary packages
using FFTW
using TensorOperations
using Plots
using Statistics

#---------------------------------------------------------------------------------#
# Necessary wrapper functions
# 1. kronecker delta value
delta_func = (x, y) -> x == y ? 1.0 : 0.0

# 2. double contradiction (4th order and 2nd order tensors)
doubledot = (A, B) -> @tensor result[a, b] := A[a, b, c, d] * B[c, d]

#3. vector and tensor dot product
dot_vec_tens = (v, T) -> @tensor res[j] := v[i] * T[i, j]


#---------------------------------------------------------------------------------#
# Inputs to the FFT algorithm
dim = 2 # number of dimension of the microstructure (2 dimension)
ngrid_1 = 129 # number of pixels in direction 1
ngrid_2 = 129 # number of pixels in direction 2
ngrids = ngrid_1 * ngrid_2 # total number of grids
lambda1, lambda2, mu1, mu2 = 0.6, 1.0, 0.6, 1.0  # lambda and mu
E_prescribed = zeros(dim, dim)
E_prescribed[1,1] = 0.1

#---------------------------------------------------------------------------------#
# Fill freq coordinates of each pixel
if ngrid_1%2 !=0
    freq_temp_1 = [-(ngrid_1-1)/2 + i for i in 0:ngrid_1-1]
else 
    freq_temp_1 = [-(ngrid_1/2) + 1 + i for i in 0:ngrid_1-1]
end

if ngrid_2%2 !=0
    freq_temp_2 = [-(ngrid_2-1)/2 + i for i in 0:ngrid_2-1]
else 
    freq_temp_2 = [-(ngrid_2/2) + 1 + i for i in 0:ngrid_2-1]
end
freq = [[freq_temp_1[i], freq_temp_2[j]] for i in 1:length(freq_temp_1), j in 1:length(freq_temp_2)]

# shifted frequencies
shifted_freq1 = fftfreq(ngrid_1, ngrid_1)
shifted_freq2 = fftfreq(ngrid_2, ngrid_2)
shifted_freq = [[shifted_freq1[i], shifted_freq2[j]] for i in 1:length(shifted_freq1), j in 1:length(shifted_freq2)]

#---------------------------------------------------------------------------------#
# Form Green operator
green_oper = zeros(ngrid_1,ngrid_2,dim,dim,dim,dim)
lambda0 = (lambda1 + lambda2)/2
mu0 = (mu1 + mu2)/2
temp_const = (lambda0 + mu0)/(mu0*(lambda0 + 2*mu0))
for y in 1:ngrid_2, x in 1:ngrid_1
    for i in 1:dim, j in 1:dim, k in 1:dim, l in 1:dim
        if (shifted_freq[x,y]' * shifted_freq[x,y]) != 0.0
            green_oper[x,y,i,j,k,l] = ((1/(4*mu0* (shifted_freq[x,y]' * shifted_freq[x,y]))) * 
                                    (delta_func(k,i)*shifted_freq[x,y][l]*shifted_freq[x,y][j] +
                                    delta_func(l,i)*shifted_freq[x,y][k]*shifted_freq[x,y][j] +
                                    delta_func(k,j)*shifted_freq[x,y][l]*shifted_freq[x,y][i] +
                                    delta_func(l,j)*shifted_freq[x,y][k]*shifted_freq[x,y][i])) -
                                    (temp_const * ((shifted_freq[x,y][i]*shifted_freq[x,y][j]*shifted_freq[x,y][k]*shifted_freq[x,y][l])
                                    /(shifted_freq[x,y]' * shifted_freq[x,y])^2))
        end
    end
end

#---------------------------------------------------------------------------------#
# Form c(x) moduli tensor

# calculate length of one pixel (between 2nd and 1st pixel): 
len_pixel = sqrt(((2-1)/ngrid_1)^2)

# calculate the radius of the inclusion
vol_frac_fiber = 0.2
lambda_of_x = zeros(ngrid_1, ngrid_2)
mu_of_x = zeros(ngrid_1, ngrid_2)
radius_fiber = sqrt((vol_frac_fiber * ngrid_1 * ngrid_2 * len_pixel^2)/π)

# form the lambda and mu tensors
center_pixel_coord_x = (div(ngrid_1,2)-1)/ngrid_1
center_pixel_coord_y = (div(ngrid_2,2)-1)/ngrid_2
for y in 1:ngrid_2, x in 1:ngrid_1
    if (((x-1)/ngrid_1 - center_pixel_coord_x)^2 + ((y-1)/ngrid_2 - center_pixel_coord_y)^2) <= (radius_fiber)^2
        lambda_of_x[x,y] = lambda2
        mu_of_x[x,y] = mu2
    else
        lambda_of_x[x,y] = lambda1
        mu_of_x[x,y] = mu1
    end
end

C_stiff = zeros(ngrid_1, ngrid_2, dim, dim, dim, dim) 
for y in 1:ngrid_2, x in 1:ngrid_1
    for i in 1:dim, j in 1:dim, k in 1:dim, l in 1:dim  
        C_stiff[x,y,i,j,k,l] = lambda_of_x[x,y] * delta_func(i,j) * delta_func(k,l) +
                                mu_of_x[x,y] *((delta_func(i,k) * delta_func(j,l)) + (delta_func(i,l) * delta_func(k,j)))
    end
end

#---------------------------------------------------------------------------------#
# Discrete algorithm

# initialization
zero_freq_grid1, zero_freq_grid2 = 1 , 1
epsilon = zeros(ngrid_1, ngrid_2, dim, dim)
sigma = zeros(ngrid_1, ngrid_2, dim, dim)
sigma_fourier = zeros(ComplexF64, ngrid_1, ngrid_2, dim, dim)
epsilon_fourier = zeros(ComplexF64, ngrid_1, ngrid_2, dim, dim)

# initialize zero_th iteration values for epsilon and sigma
for y in 1:ngrid_2, x in 1:ngrid_1
    epsilon[x,y,:,:] .= E_prescribed[:,:]
    sigma[x,y,:,:] = doubledot(C_stiff[x,y,:,:,:,:], epsilon[x,y,:,:])
end

# function to calculate the denominator value of the residuum
function calculate_denominator(sigma_fourier, p1, p2)
    result = 0
    for i in 1:dim, j in 1:dim
        result+= abs(sigma_fourier[p1,p2,i,j])^2
    end

    return sqrt(result)
end

# function to calculate the numerator value of the residuum
function calculate_numerator(sigma_fourier, freq)
    result = 0
    for y in 1:ngrid_2, x in 1:ngrid_1
        temp_result = 0
        temp = dot_vec_tens(freq[x,y], sigma_fourier[x,y,:,:])
        for i in 1:dim
            temp_result+= abs(temp[i])^2
        end
        result+=temp_result
    end
    result = sqrt(result/ngrids)
    return result
end

iter, max_iter = 1, 1000
tolerance = 1.e-8

# Loop until convergence
while true

    for i in 1:dim, j in 1:dim
        sigma_fourier[:,:,i,j] .= fft(sigma[:,:,i,j])
    end

    temp1 = calculate_numerator(sigma_fourier, shifted_freq)
    temp2 = calculate_denominator(sigma_fourier, zero_freq_grid1, zero_freq_grid2)
    println(temp1/temp2)
    if temp1/temp2 < tolerance
        break
    end
    
    for i in 1:dim, j in 1:dim
        epsilon_sum = zeros(ComplexF64, ngrid_1, ngrid_2)
        for k in 1:dim, l in 1:dim
            epsilon_sum[:,:] .+= green_oper[:,:,i,j,k,l] .* sigma_fourier[:,:,k,l]
        end
        epsilon_fourier[:,:,i,j] .= fft(epsilon[:,:,i,j]) .- epsilon_sum[:,:]
    end

    for i in 1:dim, j in 1:dim
        epsilon_fourier[zero_freq_grid1,zero_freq_grid2,i,j] = E_prescribed[i,j] 
        epsilon[:,:,i,j] .= real(ifft(epsilon_fourier[:,:,i,j]))
    end

    for i in 1:dim, j in 1:dim
        sigma_sum = zeros(ComplexF64, ngrid_1, ngrid_2)
        for k in 1:dim, l in 1:dim
            sigma_sum[:,:] .+= C_stiff[:,:,i,j,k,l] .* epsilon[:,:,k,l]
        end
        sigma[:,:,i,j] .= sigma_sum[:,:]
    end

    if iter >= max_iter
        println("Maximum iteration reached")
        println(iter)
        break
    end

    global iter = iter + 1

end

if iter != max_iter
    println("\nTotal Number of Iterations: ", iter)
end

# find the average sigma value of all pixels
function average_stress(sigma)
    _, _, dim3, dim4 = size(sigma)
    sigma_average = zeros(dim3, dim4)
    for i in 1:dim3, j in 1:dim4
        sigma_average[i,j] = mean(sigma[:,:,i,j])
    end
    return sigma_average
end
println(average_stress(sigma))

#---------------------------------------------------------------------------------#
# plot each component of sigma (stress)

# Define grid dimensions
x = range(0, stop=1, length=ngrid_1)
y = range(0, stop=1, length=ngrid_2)

# Create a contour plot function
function contour_plot_sigma(sigma, component, title)
    contour(x, y, sigma[:, :, component[1], component[2]]', xlabel="x", ylabel="y", 
            title=title, levels=50, fill=true)
end

# Plot the different components of sigma
plot1 = contour_plot_sigma(sigma, (1, 1), "sigma11")
plot2 = contour_plot_sigma(sigma, (1, 2), "sigma12")
plot3 = contour_plot_sigma(sigma, (2, 1), "sigma21")
plot4 = contour_plot_sigma(sigma, (2, 2), "sigma22")

# Arrange plots in a grid layout
plot(plot1, plot2, plot3, plot4, layout=(2, 2), size=(1000, 1000), margin=8Plots.mm)
