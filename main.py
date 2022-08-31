# Import the car_sales_dataset.csv file (it is in csv format, and the first row is the header)
# Format:
# Customer_Name,Customer_Email,Country,Gender,Age,Annual_Salary,Credit_Card_Debt,Net_Worth,Purchase_Amount
# Martina Avila,cubilia.Curae.Phasellus@quisaccumsanconvallis.edu,Bulgaria,0,41.8517198,62812.09301,11609.38091,238961.2505,35321.45877

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz
import tkinter


def main():
    df = pd.read_csv("car_sales_dataset.csv", encoding="ISO-8859-1", engine="python")

    # Identify the columns that we want to use

    # inputs = df.drop(['Age', 'Annual_Salary', 'Net_Worth', 'Purchase_Amount'], axis = 1)
    inputs = df.drop(
        ["Customer_Name", "Customer_Email", "Country", "Purchase_Amount"], axis=1
    )

    output = df["Purchase_Amount"].values.reshape(-1, 1)

    # Scale inputs and output to be between 0 and 1
    scaler_in = MinMaxScaler()
    scaler_out = MinMaxScaler()
    inputs = scaler_in.fit_transform(inputs)
    output = scaler_out.fit_transform(output)

    # Create model
    model = Sequential()
    model.add(Dense(25, input_dim=5, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(1, activation="linear"))
    print(model.summary())

    # Train model
    model.compile(loss="mean_squared_error", optimizer="adam")
    epochs_list = model.fit(
        inputs,
        output,
        epochs=100,
        batch_size=10,
        verbose=1,
        validation_split=0.2,
    )

    # # Plot the data
    # plt.plot(epochs_list.history["loss"])
    # plt.plot(epochs_list.history["val_loss"])
    # plt.title("Model Loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.legend(["Train", "Test"], loc="upper right")
    # plt.show()

    # # Visualize sequential model using ann_visualizer
    # ann_viz(model, title="Car Sales Model")

    def get_sample_data_from_user():
        """Use Tkinter"""
        root_entry: tkinter.Tk = tkinter.Tk()
        root_entry.title("Car Sales Model")
        root_entry.geometry("300x200")

        # Create a label for the input fields
        label_one = tkinter.Label(root_entry, text="Number 1")
        label_one.grid(row=0, column=0)

        label_two = tkinter.Label(root_entry, text="Number 2")
        label_two.grid(row=1, column=0)

        label_three = tkinter.Label(root_entry, text="Number 3")
        label_three.grid(row=2, column=0)

        label_four = tkinter.Label(root_entry, text="Number 4")
        label_four.grid(row=3, column=0)

        label_five = tkinter.Label(root_entry, text="Number 5")
        label_five.grid(row=4, column=0)

        # Create input fields
        input_one = tkinter.Entry(root_entry)
        input_one.grid(row=0, column=1)

        input_two = tkinter.Entry(root_entry)
        input_two.grid(row=1, column=1)

        input_three = tkinter.Entry(root_entry)
        input_three.grid(row=2, column=1)

        input_four = tkinter.Entry(root_entry)
        input_four.grid(row=3, column=1)

        input_five = tkinter.Entry(root_entry)
        input_five.grid(row=4, column=1)

        def submit():
            # Get the input fields
            df_input = np.array(
                [
                    [
                        input_one.get(),
                        input_two.get(),
                        input_three.get(),
                        input_four.get(),
                        input_five.get(),
                    ]
                ]
            )

            root_entry.destroy()

            # Scale the input fields
            df_input = scaler_in.transform(df_input)

            # Make a prediction
            prediction = model.predict(df_input)
            prediction = scaler_out.inverse_transform(prediction)

            # Pop up a gui with the prediction
            root = tkinter.Tk()
            root.title("Car Sales Model")
            root.geometry("300x200")

            # Create a label for the prediction
            label_prediction = tkinter.Label(root, text="Prediction")

            # Create a label for the prediction
            label_prediction_value = tkinter.Label(root, text=prediction)

            # Place the label and prediction on the window
            label_prediction.grid(row=0, column=0)
            label_prediction_value.grid(row=0, column=1)

            # Start the main loop
            root.mainloop()

        # Create a button to submit the input fields
        submit_button = tkinter.Button(root_entry, text="Submit", command=submit)
        submit_button.grid(row=5, column=1)

        # Start the main loop
        root_entry.mainloop()

    get_sample_data_from_user()

    # # Create sample input data and predict output
    # input_test_sample = np.array([[0, 41.8, 62812.09, 11609.38, 238961.25]])

    # # Scale input test sample data
    # input_test_sample_scaled = scaler_in.transform(input_test_sample)

    # # Output prediction
    # output_predict_sample_scaled = model.predict(input_test_sample_scaled)

    # # Print output prediction
    # print("Predicted Output (Scaled) =", output_predict_sample_scaled)

    # # Un-scale output
    # output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
    # print("Predicted Output / Purchase Amount ", output_predict_sample)


if __name__ == "__main__":
    main()
