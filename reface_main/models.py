from functions.dependency_imports import *

def landmark_directory_path(instance, filename):
    return 'user_{}/landmark/{}'.format(instance.user_code, filename)

def rebuild_part_directory_path(instance, filename):
    return 'user_{}/rebuild/{}'.format(instance.user_code, filename)

def mask_part_directory_path(instance, filename):
    return 'user_{}/mask/{}'.format(instance.user_code, str(datetime.now())+filename)
def stroke_part_directory_path(instance, filename):
    return 'user_{}/stroke/{}'.format(instance.user_code, str(datetime.now())+filename)

def origin_picture_directory_path(instance, filename):
    return 'user_{}/origin/{}'.format(instance.user_code, str(datetime.now())+filename)


class User(AbstractBaseUser, models.Model):
    user_code = models.AutoField(primary_key=True)
    account = models.CharField(max_length=30, default='')

    landmark_file = models.FileField(
        upload_to=landmark_directory_path, default=None, blank=True)
    rebuild_file = models.FileField(
        upload_to=rebuild_part_directory_path, default=None, blank=True)
    mask_file = models.FileField(
        upload_to=mask_part_directory_path, default=None, blank=True)
    stroke_file = models.FileField(
        upload_to=stroke_part_directory_path, default=None, blank=True)
    origin_file = models.FileField(
        upload_to=origin_picture_directory_path, default=None, blank=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'password' in kwargs:
            self.set_password(kwargs['password'])

    def __str__(self):
        return self.user_code
