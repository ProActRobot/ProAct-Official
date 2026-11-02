from sklearn.preprocessing import StandardScaler


def fit_and_standardize(data):
    shape = data.shape
    print(f"fit_and_standarize data.shape: {data.shape}")
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler


def standardize(data, scaler):
    shape = data.shape
    print(f"standarize data.shape: {data.shape}")
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))  # StandardScaler expected <= 2
    scaled = scaler.transform(flat).reshape(shape)
    return scaled


def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled
