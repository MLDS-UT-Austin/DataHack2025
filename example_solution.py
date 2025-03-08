# this example uses python 3.12.8, pandas, and matplotlib

# feel free to organize your repo as desired
# just don't add/remove files from the submisison folder
import pandas as pd
import matplotlib.pyplot as plt

# This function reads the input data from the provided file and saves it in a pandas dataframe. It prints the top five inputs
# for you to see
def read_input_data(input_file):
    data = pd.read_csv(input_file)
    return data

# This is a sample model. You should look into what other models are doing and import or implement those.
class WindSPModel:
    # Used to train the model
    def fit(self, X, y):
        self.__output_value = y.mean()

    # Returns the wind speed for specified number of hours
    def predict(self, X, num_hours):
        return [X['windspeed'].mean() for i in range(num_hours)]
    
# This is a sample model that outputs the average of the damage in the training data. 
# You should look into what other models are doing and import or implement those.
class DamageModel:
    # Used to train the model
    def fit(self, X, y):
        self.__output_value = y.mean()

    # Used to predict output
    def predict(self, X):
        return self.__output_value

def build_wind_speed_model(data):
    wind_speed_model = WindSPModel()
    #When fitting your model, remember that you are given a lot of training data but for the testing, you are only given 
    #five days including the wind speed and need to determine the next five days where there is no pressure or temperature.
    # You should determine whether to predict temperature, pressure, and wind speed and just output the wind speed or 
    # if you should just predict the wind speed without predicting additional columns like temperature. 
    # Some time series models let you predict multiple outputs at once so consider using those.
    wind_speed_model.fit(data[['pressure', 'air_temp', 'ground_temp']], data['windspeed'])
    return wind_speed_model

def build_damage_model(data):
    damage_model = DamageModel()
    damage_model.fit(data[['windspeed']], data['damage'])
    return damage_model

def plot_data(X, y):
    plt.scatter(X, y)
    plt.xlabel('Wind Speed')
    plt.ylabel('Damage')
    plt.title('Wind Speed vs Damage')
    plt.show()

# This should be a baseline model that you compare your code against. It should not be what you use. 
# The goal is to take in the pressure, air temperature, and ground temperature for five days and predict the wind speed for GANopolis
# for the next five days. You also need to predict the total damage to GANopolis. This model outputs the average wind speed in GANopolis for each
# day. For damage, it also looks at the average damage for the provided wind speed. We would recommend you start
# looking into other time series models such as VARIMA, as well as regression models like linear regression and random forest.
def main():
    #We only need the information about GANopolis so only get its information
    data = read_input_data('data/training_data.csv')
    ganopolis_information = data[data['city'] == 'GANopolis'].sort_values(by=['hours', 'hour_of_day'])
    print(ganopolis_information.head())

    wind_speed_model = build_wind_speed_model(data)
    
    #We should look at the relationship between wind speed and damage. Let us plot the values and see what relationship there is.
    plot_data(data['windspeed'], data['damage'])
    damage_model = build_damage_model(data)

    # We can now predict the wind speed and damage for the sample data. The file gives five days of data and we predict the next five days.
    # We have ten events so we need to predict ten different outputs.
    output = []

    for event_number in range(1, 11, 1):
        prediction_data = read_input_data('data/event_'+str(event_number)+'.csv')
        ganopolis_prediction = prediction_data.loc[data['city'] == 'GANopolis'].sort_values(by=['hour', 'hour_of_day'])
        # We get the wind speed for the number of hours
        num_hours = 120
        wind_speed = wind_speed_model.predict(ganopolis_prediction[['pressure', 'air_temp', 'ground_temp', 'windspeed']], num_hours)
        
        # The damage is damage for the five days you predict.
        totalDamage = 0
        for i in range(num_hours):
            totalDamage = totalDamage + damage_model.predict(wind_speed[i])

        # This is optimal price for a given damage, derived using the derivative of the profit formula
        price = 250 + totalDamage / 2

        output_dict = {
            "event_number": event_number,
            "price": price,
            **{str(i): wind_speed[i] for i in range(num_hours)},
        }
        output.append(pd.DataFrame([output_dict]))

    output_df = pd.concat(output, ignore_index=True)
    output_df.to_csv('submission/submission.csv', index=False)

main()