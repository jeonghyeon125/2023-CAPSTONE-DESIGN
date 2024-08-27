from rest_framework import serializers
from .models import InOut, Video, STT

class InoutSerializer(serializers.ModelSerializer):
    class Meta:
        model = InOut
        fileds = ('URL', 'similarity')

    def to_representation(self, instance):
        self.fields['URL'] = InoutRepresentationSerializer(read_only = True)
        return super(InoutSerializer, self).to_representation(instance)

class VideoSerializer(serializers.ModelSerializer):
    in_vid = InoutSerializer(many = True)

    class Meta:
        model = Video
        fileds = ('URL', 'vid')

class STTSerializer(serializers.ModelSerializer):
    in_stt = InoutSerializer(many = True)

    class Meta:
        model = STT
        fileds = ('URL', 'stt_result')

class InoutRepresentationSerializer(serializers.ModelSerializer):
    class Meta:
        model = InOut
        fields = ('URL', 'stt_result')