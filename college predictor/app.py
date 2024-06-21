import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
from flask import Flask, render_template, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
import bcrypt
from flask_mysqldb import MySQL

app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'shailesh'
app.config['MYSQL_DB'] = 'shailesh'
app.secret_key = 'your_secret_key_here'

mysql = MySQL(app)

class RegisterForm(FlaskForm):
    name = StringField("Name",validators=[DataRequired()])
    email = StringField("Email",validators=[DataRequired(), Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self,field):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users where email=%s",(field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError('Email Already Taken')

class LoginForm(FlaskForm):
    email = StringField("Email",validators=[DataRequired(), Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    submit = SubmitField("Login")



# rendering pages 
@app.route("/")
def home():
    return render_template("homepage.html")


@app.route('/register',methods=['GET','POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data

        hashed_password = bcrypt.hashpw(password.encode('utf-8'),bcrypt.gensalt())

        # store data into database 
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (name,email,password) VALUES (%s,%s,%s)",(name,email,hashed_password))
        mysql.connection.commit()
        cursor.close()

        return redirect(url_for('home'))

    return render_template('register.html',form=form)

@app.route('/login',methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s",(email,))
        user = cursor.fetchone()
        cursor.close()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            session['user_id'] = user[0]
            return redirect(url_for('home'))
        else:
            flash("Login failed. Please check your email and password")
            return redirect(url_for('login'))

    return render_template('login.html',form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))


@app.route("/explore_branches")
# @login_required
def explore_branches():
    return render_template("explore_branches.html")

@app.route("/about")
# @login_required
def about():
    return render_template("about.html")

@app.route("/feedback")
# @login_required
def feedback():
    return render_template("feedback.html")










# Load your dataset
df = pd.read_csv('cleaned_data.csv')  

# Perform label encoding
le_category = LabelEncoder()
le_quota = LabelEncoder()
le_branch = LabelEncoder()
le_location = LabelEncoder()
le_college = LabelEncoder()

df['category'] = le_category.fit_transform(df['category'])
df['quota'] = le_quota.fit_transform(df['quota'])
df['Branch'] = le_branch.fit_transform(df['Branch'])
df['location'] = le_location.fit_transform(df['location'])
df['College Name'] = le_college.fit_transform(df['College Name'])


X = df[['round_no', 'Branch', 'quota', 'category', 'Closing_CutOff', 'location']]
y = df['College Name']

decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X, y)

num_clusters = 70  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

@app.route('/form', methods=['GET', 'POST'])

def form():
    if request.method == 'POST':
        round_no = int(request.form['round_no'])
        branch = request.form['branch']
        quota = request.form['quota']
        category = request.form['category']
        closing_cut_off = int(request.form['closing_cut_off'])
        location = request.form['location']

        input_data = [[round_no, le_branch.transform([branch])[0], le_quota.transform([quota])[0],
                       le_category.transform([category])[0], closing_cut_off, le_location.transform([location])[0]]]

        prediction = decision_tree_classifier.predict(input_data)
        cluster_prediction = kmeans.predict(input_data)[0]

        
        unique_colleges = df[df['Cluster'] == cluster_prediction].groupby('College Name').first().reset_index()

        recommended_colleges = le_college.inverse_transform(unique_colleges['College Name'])[:70]

        return render_template('result.html', predicted_college_name=le_college.inverse_transform([prediction[0]])[0],
                               recommended_colleges=recommended_colleges)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)









