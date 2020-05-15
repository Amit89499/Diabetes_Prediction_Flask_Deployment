from flask import Flask, request, render_template
#import pickle
import numpy as np
from sklearn.externals import joblib
app=Flask(__name__)

@app.route('/')

def index():
	return render_template('index.html')

@app.route('/prediction',methods=['POST'])

def prediction():
	Pregnancies=request.form.get("pregnancies")
	glucose=request.form.get("glucose")
	bloodPressure=request.form.get("bloodPressure")
	SkinThickness=request.form.get("SkinThickness")
	Insulin=request.form.get("Insulin")
	bmi=request.form.get("bmi")
	DiabetesPedigreeFunction=request.form.get("DiabetesPedigreeFunction")
	age=request.form.get("age")
	Output=request.form.get("Output")


	Pregnancies=int(Pregnancies)
	glucose=int(glucose)
	bloodPressure=int(bloodPressure)
	SkinThickness=int(SkinThickness)
	Insulin=int(Insulin)
	bmi=float(bmi)
	DiabetesPedigreeFunction=float(DiabetesPedigreeFunction)
	age=int(age)
	Output=int(Output)
	x_factors=[[Pregnancies,glucose,bloodPressure,SkinThickness,Insulin,bmi,DiabetesPedigreeFunction,age,Output]]
	#x_factors=x_factors.reshape(-1,1)
	# prediction=predictors.reshape(-1,1)
	# model=pickle.load(open('model.pkl','rb'))
	# predict = model.predict(prediction)
	model = joblib.load('model.pkl')
	prediction = model.predict(x_factors)
	# return render_template('result.html', prediction = prediction)
	# prediction=prediction.ravel()
	return render_template('result.html',prediction=prediction)
	"""except Exception as e:

		print('The Exception message is: ',e)
		print("Something went wrong")"""

if __name__ == '__main__':
	app.run(debug=True)