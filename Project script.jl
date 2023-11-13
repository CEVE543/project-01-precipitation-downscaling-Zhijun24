#Temporary note: example code and data  will be pushed ASAP but is not yet live.

#In this project, you will build on recent labs to develop a downscaling model. Specifically, you will build a model to downscale precipitation, temperature, and other  fields from ERA5 reanalysisLinks to an external site. to NEXRAD radar precipitationLinks to an external site. over Houston. Within these bounds, you are free to define your problem (are you trying to predict a high-resolution rainfall field at time t+1 given all available information at time t? Are you trying to estimate local daily precipitation from regional climate variables? Are you trying to map hourly high-resolution precipitation from hourly low-resolution precipitation?) and to choose your own methods.

#This is a class project, not a detailed report, so define your problem early and work with what you have!


using CSV
using DataFrames
using DataFramesMeta
using Dates
using Plots
using NCDatasets
using MultivariateStats
using Unitful
using StatsBase
using StatsPlots

#read precipitation data
precip_ds = NCDataset("data/precip_tx.nc") #1979 to 2023
precip_time = precip_ds["time"][:] #16365-element
precip_lon = precip_ds["lon"][:] #24-element
precip_lat = precip_ds["lat"][:] #24-element
precip_lat = reverse(precip_lat)
precip = precip_ds["precip"][:, :, :] .* 1.0u"mm" #24*24*16365
precip = reverse(precip; dims=2)



#read temperature data
#1. get longitude and latitude information
temp_ds = NCDataset("data/raw/2m_temperature_1979.nc")
temp_lon = temp_ds["longitude"][:] #12-element
temp_lat = temp_ds["latitude"][:] #12-element
temp_lat = reverse(temp_lat) #reverse the latitude data

#2. get temperature data from year 1979 to 2022
temp = []
temp_time = []
for i in 1979:2022 #1979 to 2022
    temp_i_ds = NCDataset("data/raw/2m_temperature_$i.nc")
    temp_i_time = temp_i_ds["time"][:]
    temp_i = temp_i_ds["t2m"][:, :, :] .* 1.0u"K"
    if i == 1979
        temp_time = temp_i_time
        temp = temp_i
    else
        temp_time = vcat(temp_time, temp_i_time)
        temp = cat(temp, temp_i, dims=3)
    end
end
temp = reverse(temp; dims=2)

#read pressure data
#1. get longitude and latitude data 
pressure_i_ds = NCDataset("data/raw/500hPa_geopotential_1979.nc")
pressure_lon = pressure_i_ds["longitude"][:] #12-element
pressure_lat = pressure_i_ds["latitude"][:] #12-element
pressure_lat = reverse(pressure_lat)
#2. get pressure data from year 1979 to 2022
pressure = []
pressure_time = []
for i in 1979:2022 #1979 to 2022
    pressure_i_ds = NCDataset("data/raw/500hPa_geopotential_$i.nc")
    pressure_i_time = pressure_i_ds["time"][:]
    pressure_i = pressure_i_ds["z"][:, :, :] .* 1.0u"hPa"
    if i == 1979
        pressure_time = pressure_i_time
        pressure = pressure_i
    else
        pressure_time = vcat(pressure_time, pressure_i_time)
        pressure = cat(pressure, pressure_i, dims=3)
    end
end
pressure = reverse(pressure; dims=2)

#temperature data for a single day
P_temp = heatmap(
    temp_lon, # x labels
    temp_lat, # y labels
    temp[:, :, 15000]';
    size=(600, 700),
    title="Temperature on $(temp_time[15000])",
    xlabel="Longitude",
    ylabel="Latitude"
)

#pressure data for a single day
P_pressure = heatmap(
    pressure_lon, # x labels
    pressure_lat, # y labels
    pressure[:, :, 15000]';
    size=(600, 700),
    title="Pressure on $(pressure_time[1])",
    xlabel="Longitude",
    ylabel="Latitude"
)

#precipitation data for a single day
P_precip = heatmap(
    precip_lon, # x labels
    precip_lat, # y labels
    precip[:, :, 15000]';
    size=(600, 700),
    title="Precipitation on $(precip_time[15000])",
    xlabel="Longitude",
    ylabel="Latitude"
)

time = temp_time

#split the data; use the first 10 month as train data and last 2 month as test data
idx_partition = findfirst(time .== time[end] - Dates.Year(5))
train_idx = 1:idx_partition
test_idx = (idx_partition+1):length(time)
time_train = time[1:idx_partition]
time_test = time[idx_partition+1:length(time)]
precip_train = precip[:, :, train_idx]
precip_test = precip[:, :, test_idx]
temp_train = temp[:, :, train_idx]
temp_test = temp[:, :, test_idx]
pressure_train = pressure[:, :, train_idx]
pressure_test = pressure[:, :, test_idx]

#scatter plot for precipitation
avg_precip =
    ustrip.(
        u"inch", [mean(skipmissing(precip_train[:, :, t])) for t in 1:size(precip_train, 3)]
    )
p_precip_avg = scatter(
    time[1:idx_partition],
    avg_precip;
    xlabel="Day",
    ylabel="Precipitation",
    markersize=5,
    clims=(0, 0.5),
    title="Average precipitation",
    label=false
)



#preprocess
function preprocess(temp::Array{T,3}, temp_ref::Array{T,3})::AbstractMatrix where {T}
    n_lon, n_lat, n_t = size(temp)
    climatology = mean(temp_ref; dims=3)
    temp_anom = temp .- climatology
    temp_anom = reshape(temp_anom, n_lon * n_lat, n_t)
    return ustrip.(temp_anom)
end

temp_mat_train = preprocess(temp_train, temp_train)
temp_mat_test = preprocess(temp_test, temp_test)

pressure_mat_train = preprocess(pressure_train, pressure_train)
pressure_mat_test = preprocess(pressure_test, pressure_test)

#PCA
#fitting
pca_model_temp = fit(PCA, temp_mat_train; maxoutdim=25, pratio=0.999)
pca_model_pressure = fit(PCA, pressure_mat_train; maxoutdim=25, pratio=0.999)
#plot
p1 = plot(
    principalvars(pca_model_temp) / var(pca_model_temp);
    xlabel="number of PCs",
    ylabel="Fraction of Variance Explained",
    label=false,
    title="Variance Explained"
)
p2 = plot(
    cumsum(principalvars(pca_model_temp)) / var(pca_model_temp);
    xlabel="number of PCs",
    ylabel="Fraction of Variance Explained",
    label=false,
    title="Cumulative Variance Explained"
)
plot(p1, p2; layout=(1, 2), size=(800, 400))

#plot of projection
p = []
n_lon, n_lat, n_t = size(temp)
for i in 1:3
    pc = projection(pca_model_temp)[:, i]
    pc = reshape(pc, n_lat, n_lon)'
    pi = heatmap(
        temp_lon,
        temp_lat,
        pc;
        xlabel="Longitude",
        ylabel="Latitude",
        title="PC $i",
        aspect_ratio=:equal,
        cmap=:PuOr
    )
    push!(p, pi)
end
plot(p...; layout=(1, 3), size=(1500, 600))



pc_ts_temp = predict(pca_model_temp, temp_mat_train)
day_of_year = Dates.dayofyear.(time_train)
p = []
for i in 1:3
    pi = scatter(
        day_of_year,
        pc_ts_temp[i, :];
        xlabel="Day of Year",
        ylabel="PC $i",
        title="PC $i",
        label=false,
        alpha=0.3,
        color=:gray
    )
    push!(p, pi)
end
plot(p...; layout=(1, 3), size=(1500, 600))


avg_precip =
    ustrip.(
        u"mm", [mean(skipmissing(precip_train[:, :, t])) for t in 1:size(precip_train, 3)]
    )
avg_precip = replace(avg_precip, NaN => 0)
p2_idx = findall(avg_precip .> quantile(avg_precip, 0.98))
p1 = scatter(
    pc_ts_temp[2, :],
    pc_ts_temp[3, :];
    zcolor=avg_precip,
    xlabel="PC 2",
    ylabel="PC 3",
    markersize=3,
    clims=(0, maximum(avg_precip[p2_idx])),
    title="All Days",
    label=false
)


p2 = scatter(
    pc_ts_temp[2, p2_idx],
    pc_ts_temp[3, p2_idx];
    zcolor=avg_precip[p2_idx],
    xlabel="PC 2",
    ylabel="PC 3",
    markersize=5,
    clims=(0, maximum(avg_precip[p2_idx])),
    title="Rainy Days",
    label=false
)
plot(p1, p2; size=(1000, 400), link=:both)

#knn
function euclidean_distance(x::AbstractVector, y::AbstractVector)::AbstractFloat
    return sqrt(sum((x .- y) .^ 2))
end
function nsmallest(x::AbstractVector, n::Int)::Vector{Int}
    idx = sortperm(x)
    return idx[1:n]
end
function knn(X::AbstractMatrix, X_i::AbstractVector, K::Int)::Tuple{Int,AbstractVector}
    # calculate the distances between X_i and each row of X
    dist = [euclidean_distance(X_i, X[j, :]) for j in 1:size(X, 1)]
    idx = nsmallest(dist, K)
    w = 1 ./ dist[idx]
    w ./= sum(w)
    idx_sample = sample(idx, Weights(w))
    return (idx_sample, vec(X[idx_sample, :]))
end

#pca and knn
function predict_knn(temp_train, temp_test, precip_train; n_pca::Int)
    X_train = preprocess(temp_train, temp_train)
    X_test = preprocess(temp_test, temp_train)
    # fit the PCA model to the training data
    pca_model_temp = fit(PCA, X_train; maxoutdim=n_pca)
    # project the test data onto the PCA basis
    train_embedded = predict(pca_model_temp, X_train)
    test_embedded = predict(pca_model_temp, X_test)
    # use the `knn` function for each point in the test data
    precip_pred = map(1:size(X_test, 2)) do i
        idx, _ = knn(train_embedded', test_embedded[:, i], 3)
        precip_train[:, :, idx]
    end
    # return a matrix of predictions
    return precip_pred
end

#precipitation prediction based on the temperature
t_sample_temp = rand(1:size(temp_test, 3), 3)
precip_pred_temp = predict_knn(temp_train, temp_test[:, :, t_sample_temp], precip_train; n_pca=3)
p = map(eachindex(t_sample_temp)) do ti
    t = t_sample_temp[ti]
    y_pred = precip_pred_temp[ti]'
    y_actual = precip_test[:, :, t]'
    cmax = max(maximum(skipmissing(y_pred)), maximum(skipmissing(y_actual)))
    cmax = ustrip(u"mm", cmax)
    p1 = heatmap(
        precip_lon,
        precip_lat,
        y_pred;
        xlabel="Longitude",
        ylabel="Latitude",
        title="Predicted",
        aspect_ratio=:equal,
        clims=(0, cmax)
    )
    p2 = heatmap(
        precip_lon,
        precip_lat,
        y_actual;
        xlabel="Longitude",
        ylabel="Latitude",
        title="Actual",
        aspect_ratio=:equal,
        clims=(0, cmax)
    )
    plot(p1, p2; layout=(2, 1), size=(1000, 400))
end
plot(p...; layout=(2, 3), size=(1500, 1200))

#precipitation prediction based on the pressure
t_sample_pressure = rand(1:size(pressure_test, 3), 3)
precip_pred_pressure = predict_knn(pressure_train, pressure_test[:, :, t_sample_pressure], precip_train; n_pca=3)
p = map(eachindex(t_sample_pressure)) do ti
    t = t_sample_pressure[ti]
    y_pred = precip_pred_pressure[ti]'
    y_actual = precip_test[:, :, t]'
    cmax = max(maximum(skipmissing(y_pred)), maximum(skipmissing(y_actual)))
    cmax = ustrip(u"mm", cmax)
    p1 = heatmap(
        precip_lon,
        precip_lat,
        y_pred;
        xlabel="Longitude",
        ylabel="Latitude",
        title="Predicted",
        aspect_ratio=:equal,
        clims=(0, cmax)
    )
    p2 = heatmap(
        precip_lon,
        precip_lat,
        y_actual;
        xlabel="Longitude",
        ylabel="Latitude",
        title="Actual",
        aspect_ratio=:equal,
        clims=(0, cmax)
    )
    plot(p1, p2; layout=(2, 1), size=(1000, 400))
end
plot(p...; layout=(2, 3), size=(1500, 1200))


#probability density function (pdf) of the precipitation
avg_precip_sample =
    ustrip.(
        u"mm", [mean(skipmissing(precip[:, :, t])) for t in 1:size(precip, 3)]
    )

histogram(avg_precip_sample; bins=avg_precip_sample, label="Precipitation", normalize=:pdf, xticks=1:10)

#pdf of precipitation on rainyday
histogram(avg_precip[p2_idx]; bins=avg_precip[p2_idx], label="Precipitation", normalize=:pdf, xticks=1:maximum(avg_precip[p2_idx]))

#probability density function (pdf) of the temperature
avg_temp_sample =
    ustrip.(
        u"K", [mean(skipmissing(temp[:, :, t])) for t in 1:size(temp, 3)]
    )

histogram(avg_temp_sample; bins=avg_temp_sample, label="Temperature", normalize=:pdf, xticks=minimum(avg_temp_sample):maximum(avg_temp_sample))



#read cloud cover fraction data
#1. get longitude and latitude information
cc_i_ds = NCDataset("data/raw_1/500hPa_cloud_cover_1979.nc")
cc_lon = cc_i_ds["longitude"][:] #12-element
cc_lat = cc_i_ds["latitude"][:] #12-element
cc_lat = reverse(cc_lat) #reverse the latitude data
#2. get cloud cover data from year 1979 to 2022
cc = []
cc_time = []
for i in 1979:2022 #1979 to 2022
    cc_i_ds = NCDataset("data/raw_1/500hPa_cloud_cover_$i.nc")
    cc_i_time = cc_i_ds["time"][:]
    cc_i = cc_i_ds["cc"][:, :, :]
    if i == 1979
        cc_time = cc_i_time
        cc = cc_i
    else
        cc_time = vcat(cc_time, cc_i_time)
        cc = cat(cc, cc_i, dims=3)
    end
end
cc = reverse(cc; dims=2)

#precipitation prediction based on the cloud cover
cloud_cover_train = cc[:, :, train_idx]
cloud_cover_test = cc[:, :, test_idx]
t_sample_cloud_cover = rand(1:size(cloud_cover_test, 3), 3)
precip_pred_cloud_cover = predict_knn(cloud_cover_train, cloud_cover_test[:, :, t_sample_cloud_cover], precip_train; n_pca=3)
p = map(eachindex(t_sample_cloud_cover)) do ti
    t = t_sample_cloud_cover[ti]
    y_pred = precip_pred_cloud_cover[ti]'
    y_actual = precip_test[:, :, t]'
    cmax = max(maximum(skipmissing(y_pred)), maximum(skipmissing(y_actual)))
    cmax = ustrip(u"mm", cmax)
    p1 = heatmap(
        precip_lon,
        precip_lat,
        y_pred;
        xlabel="Longitude",
        ylabel="Latitude",
        title="Predicted",
        aspect_ratio=:equal,
        clims=(0, cmax)
    )
    p2 = heatmap(
        precip_lon,
        precip_lat,
        y_actual;
        xlabel="Longitude",
        ylabel="Latitude",
        title="Actual",
        aspect_ratio=:equal,
        clims=(0, cmax)
    )
    plot(p1, p2; layout=(2, 1), size=(1000, 400))
end
plot(p...; layout=(2, 3), size=(1500, 1200))

#read dew temperature data
#1. get longitude and latitude information
d2m_i_ds = NCDataset("data/raw_1/d2m_temperature_1979.nc")
d2m_lon = d2m_i_ds["longitude"][:] #12-element
d2m_lat = d2m_i_ds["latitude"][:] #12-element
d2m_lat = reverse(d2m_lat) #reverse the latitude data
#2. get dew temperature data from year 1979 to 2022
d2m = []
d2m_time = []
for i in 1979:2022 #1979 to 2022
    d2m_i_ds = NCDataset("data/raw_1/d2m_temperature_$i.nc")
    d2m_i_time = d2m_i_ds["time"][:]
    d2m_i = d2m_i_ds["d2m"][:, :, :]
    if i == 1979
        d2m_time = d2m_i_time
        d2m = d2m_i
    else
        d2m_time = vcat(d2m_time, d2m_i_time)
        d2m = cat(d2m, d2m_i, dims=3)
    end
end
d2m = reverse(d2m; dims=2)

#precipitation prediction based on the cloud cover
d2m_train = cc[:, :, train_idx]
d2m_test = cc[:, :, test_idx]
t_sample_d2m = rand(1:size(d2m_test, 3), 3)
precip_pred_d2m = predict_knn(d2m_train, d2m_test[:, :, t_sample_d2m], precip_train; n_pca=3)
p = map(eachindex(t_sample_d2m)) do ti
    t = t_sample_d2m[ti]
    y_pred = precip_pred_d2m[ti]'
    y_actual = precip_test[:, :, t]'
    cmax = max(maximum(skipmissing(y_pred)), maximum(skipmissing(y_actual)))
    cmax = ustrip(u"mm", cmax)
    p1 = heatmap(
        precip_lon,
        precip_lat,
        y_pred;
        xlabel="Longitude",
        ylabel="Latitude",
        title="Predicted",
        aspect_ratio=:equal,
        clims=(0, cmax)
    )
    p2 = heatmap(
        precip_lon,
        precip_lat,
        y_actual;
        xlabel="Longitude",
        ylabel="Latitude",
        title="Actual",
        aspect_ratio=:equal,
        clims=(0, cmax)
    )
    plot(p1, p2; layout=(2, 1), size=(1000, 400))
end
plot(p...; layout=(2, 3), size=(1500, 1200))