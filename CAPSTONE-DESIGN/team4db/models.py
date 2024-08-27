from django.db import models

# Create your models here.

""" 사전 """
class adjective(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9')

class adverb(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9')

class dic_usr(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column2')
    note = models.TextField(null = True, db_column = 'Column3')

class eomi(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9')  

class exclamation(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9')

class foreign(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9') 

class hanja(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9')

class josa(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9')

class noun(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column2')
    note = models.TextField(null = True, db_column = 'Column3')

class suffix(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9') 

class symbol(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9')

class verb(models.Model):
    word = models.TextField(null = True, db_column = 'Column1')
    word_type = models.TextField(null = True, db_column = 'Column5')
    note = models.TextField(null = True, db_column = 'Column9')

""" 입력, 결과 """
class InOut(models.Model):
    URL = models.URLField(primary_key = True)
    similarity = models.FloatField(null = True)

""" stt """
class STT(models.Model):
    URL = models.OneToOneField(InOut, related_name = 'in_stt', on_delete = models.CASCADE)
    stt_result = models.TextField(null = True)
    cleaned = models.TextField(null = True)
    preprocessing = models.TextField(null = True)  

""" 추출된 키워드 """
class Keyword(models.Model):
    URL = models.OneToOneField(InOut, on_delete = models.CASCADE)
    keyword1 = models.TextField(null = True)
    keyword2 = models.TextField(null = True)    
    keyword3 = models.TextField(null = True)    
    keyword4 = models.TextField(null = True)    
    keyword5 = models.TextField(null = True)     

""" 학습할 네이버 뉴스 """
class train_naver(models.Model):
    URL = models.ForeignKey(InOut, on_delete = models.CASCADE)
    url_naver = models.URLField(null = True)
    title = models.TextField(null = True)
    published_date = models.TextField(null = True)
    body = models.TextField(null = True) 

""" 학습할 snu 팩트체크 """
class train_snu(models.Model):
    url_snu = models.URLField(null = True)
    title = models.TextField(null = True)
    published_date = models.TextField(null = True)
    body = models.TextField(null = True)

""" aws s3 """
from django.db import models
from django.conf import settings
from django.contrib.auth.models import User

class Video(models.Model):
    URL = models.ForeignKey(InOut, on_delete = models.CASCADE)
    vid = models.FileField(upload_to="", blank=True, unique=False)

