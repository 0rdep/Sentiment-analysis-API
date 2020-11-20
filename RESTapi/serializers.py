from rest_framework import serializers

class ClassificationPostRequestSerializer(serializers.Serializer):
    word = serializers.CharField()
    valance = serializers.FloatField()
    arousal = serializers.FloatField()
    dominance = serializers.FloatField()

class ClassificationPostSuccessResponseSerializer(serializers.Serializer):
    message = serializers.CharField()
    code = serializers.IntegerField()
    status = serializers.CharField()
    dominance = serializers.FloatField()

    
class ClassificationPostErrorResponseSerializer(serializers.Serializer):
    message = serializers.CharField()
    code = serializers.IntegerField()
    status = serializers.CharField()
    error = serializers.CharField()

class ClassificationGetResponseSerializer(serializers.Serializer):
    url = serializers.CharField()
    url = serializers.CharField()
    data = serializers.ListField(child=serializers.CharField())
    code = serializers.IntegerField()
    status = serializers.CharField()
    url = serializers.CharField()