def calculate_prediction_sales(row, columns = ["PredictionScaled", "SalesScaled"]):
    ### This function inverses the applied min max scaling
    c_min_max = {
        0: {"Min": 17.522332908595782, "Max":164477.3347483395},
        1: {"Min": -45.04917947145333, "Max": 240464.1980553548},
        2: {"Min": 9958.738964463202, "Max": 1659719.698434384}
    }
    min_value = c_min_max[row["Company"]]["Min"]
    max_value = c_min_max[row["Company"]]["Max"]
    
    dict = {}

    for i in range(len(columns)):
        column = columns[i]
        to_scale = row[column]

        scaled = to_scale * (max_value - min_value) + min_value

        dict[i] = scaled


    # predict_scaled = row["PredictionScaled"]
    # sales_scaled = row["SalesScaled"]

    # prediction = predict_scaled * (max_value - min_value) + min_value
    # sales = sales_scaled * (max_value - min_value) + min_value
    return dict