#Temporary note: example code and data  will be pushed ASAP but is not yet live.

#In this project, you will build on recent labs to develop a downscaling model. Specifically, you will build a model to downscale precipitation, temperature, and other  fields from ERA5 reanalysisLinks to an external site. to NEXRAD radar precipitationLinks to an external site. over Houston. Within these bounds, you are free to define your problem (are you trying to predict a high-resolution rainfall field at time t+1 given all available information at time t? Are you trying to estimate local daily precipitation from regional climate variables? Are you trying to map hourly high-resolution precipitation from hourly low-resolution precipitation?) and to choose your own methods.

#This is a class project, not a detailed report, so define your problem early and work with what you have!


using CSV
using DataFrames
using DataFramesMeta
using Dates
using Plots
using NCDatasets
using Unitful
using StatsBase
using StatsPlots

precip_ds = NCDataset("data/precip_tx.nc")
precip_time = precip_ds["time"][:] #16365-element
precip_lon = precip_ds["lon"][:] #24-element
precip_lat = precip_ds["lat"][:] #24-element
precip = precip_ds["precip"][:, :, :] .* 1.0u"mm" #24*24*16365

temp_ds = NCDataset("data/adaptor.mars.internal-1698079032.1625195-32438-12-7dd363d5-fdf1-4a3e-ac38-f37e1a2c0ca7.nc")
temp_time = temp_ds["time"][:] #8760-element; year 2019
temp_lon = temp_ds["longitude"][:] #66-element
temp_lat = temp_ds["latitude"][:] #27-element
temp = temp_ds["t2m"][:, :, :] .* 1.0u"K" #66*27*8760

#heatmap of temperature data for a single hour (2019-01-01T00:00:00)
temp = temp .* 1.0u"K^-1"
P_temp = heatmap(
    temp_lon, # x labels
    reverse(temp_lat), # y labels
    temp[:, :, 1][:, end:-1:1]'; 
    size=(600, 700), 
    title="Temperature",
    xlabel="longitude",
    ylabel="latitude"
)

#precipitation data (1979-01-01T00:00:00) from a single day
precip = precip .* 1.0u"mm^-1"
P_precip = heatmap(
    precip_lon, # x labels
    reverse(precip_lat), # y labels
    precip[:, :, 1][:, end:-1:1]'; 
    size=(600, 700),
    title="Precipitation",
    xlabel="longitude",
    ylabel="latitude"
)

#new temperature data input (1979-01-01T00:00:00) from a single hour
temp_1979_ds = NCDataset("data/adaptor.mars.internal-1698952099.6748757-3382-5-bbeebfb4-bada-4eee-a3b1-694449020830.nc")
temp_1979_time = temp_1979_ds["time"][:] 
temp_1979_lon = temp_1979_ds["longitude"][:] 
temp_1979_lat = temp_1979_ds["latitude"][:] 
temp_1979 = temp_1979_ds["t2m"][:, :, :] .* 1.0u"K" 

temp_1979 = temp_1979 .* 1.0u"K^-1"
P_temp_1979 = heatmap(
    temp_1979_lon, # x labels
    reverse(temp_1979_lat), # y labels
    temp_1979[:, :, 12][:, end:-1:1]'; 
    size=(600, 700), 
    title="Temperature",
    xlabel="longitude",
    ylabel="latitude"
)

#Dayly average temperature data from 2021

temp_2021_ds = NCDataset("data/adaptor.mars.internal-1698954510.7643569-18249-17-4f589d3f-51f6-4424-8979-e64d4e0db160.nc")
temp_2021_time = temp_2021_ds["time"][:] 
temp_2021_lon = temp_2021_ds["longitude"][:] 
temp_2021_lat = temp_2021_ds["latitude"][:] 
temp_2021 = temp_2021_ds["t2m"][:, :, :] .* 1.0u"K" 

#Dayly preciptiation data from 2021
precip_2021_time_idx = 14976:1:15341
precip_2021_time = precip_time[14976:15341]
precip_2021 = precip[:, :, precip_2021_time_idx]
