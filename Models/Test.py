import pandas as pd

df = pd.DataFrame(data={
    "Day": [0, 1, 2, 3, 4, 5, 6, 7],
    "Sales": [0, 10, 20, 30, 40, 50, 60, 70]
})

print(df.shift(1).rolling(2).mean())