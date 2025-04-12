from django.db import models

# Create your models here.
class manage_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'manage_users'



class LR(models.Model):
    """
    This model stores the evaluation metrics of the Linear Regression algorithm
    """
    accuracy = models.FloatField()  # Store accuracy as a floating-point number
    mse = models.FloatField()  # Mean Squared Error
    rmse = models.FloatField()  # Root Mean Squared Error

    def __str__(self):
        return f"Linear Regression Results (Accuracy: {self.accuracy:.2f}, MSE: {self.mse:.2f}, RMSE: {self.rmse:.2f})"
    class Meta:
        db_table = 'LR'



class RF(models.Model):
    """
    This model stores the evaluation metrics of the Random Forest algorithm
    """
    accuracy = models.FloatField()  # Store accuracy as a floating-point number
    mse = models.FloatField()  # Mean Squared Error
    rmse = models.FloatField()  # Root Mean Squared Error

    def __str__(self):
        return f"Random Forest Results (Accuracy: {self.accuracy:.2f}, MSE: {self.mse:.2f}, RMSE: {self.rmse:.2f})"
    class Meta:
        db_table = 'RF'



class ANN(models.Model):
    """
    This model stores the evaluation metrics of the ANN algorithm
    """

    accuracy = models.FloatField()  # Store accuracy as a floating-point number
    mse = models.FloatField()  # Mean Squared Error
    rmse = models.FloatField()  # Root Mean Squared Error

    def __str__(self):
        return f"ANN Results (Accuracy: {self.accuracy:.2f}, MSE: {self.mse:.2f}, RMSE: {self.rmse:.2f})"
    class Meta:
        db_table = 'ANN'