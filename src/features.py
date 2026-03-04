from sklearn.preprocessing import StandardScaler

def get_feature_space(df, mode="combined"):

    if mode == "photometry":
        features = df[["M_G", "color"]]

    elif mode == "kinematics":
        features = df[["parallax", "pmra", "pmdec", "v_t"]]

    elif mode == "combined":
        features = df[[
            "M_G", "color",
            "parallax", "pmra", "pmdec", "v_t"
        ]]

    features = features.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    return X_scaled, features.index