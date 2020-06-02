from django.shortcuts import render
from functions.dependency_imports import *
import functions.basefunction as base
from reface_main.serializers import *
from functions.onetake import *
# Create your views here.
@api_view(['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
def UserResponse(request, url=None, extra=None):
    if request.method == 'GET':
        authorized, authorize_object, response, request = base.Authorize_session(request)
        mode = request.get('mode')
        if authorized:
            if mode == 'inference':
                image_name = request.get('image')
                image_name = Image.open(image_name)
                image_name.save('hi.png')
                onetake_gcs(str(authorize_object.user_code),image_name,dbface=True,readdat=True)
                return base.Custom_Response(200,'done')
            elif mode == 'inference_origin':
                image_name = request.get('image')
                onetake_gcs(str(authorize_object.user_code),image_name,dbface=False,readdat=True,origin=True)
                return base.Custom_Response(200,'done')
            elif mode == 'inference_preprocess':
                preprocess_images = []
                for oneImage in request.getlist('originimage'):
                    preprocess_images.append(Image.open(oneImage))
                rebuild_image = Image.open(request.get('rebuildimage'))
                mask_image = Image.open(request.get('maskimage'))
                return preprocess(authorize_object.user_code,rebuild_image,preprocess_images,mask_image,None)

                #onetake_gcs(str(authorize_object.user_code),image_name,dbface=False,readdat=True,preprocess=True)
                return base.Custom_Response(200,'done')
            else:
                return base.Custom_Response(502,'not implemented')
        else:
            return response
    elif request.method == 'POST':
        authorized, authorize_object, response, request = base.Authorize_session(request)
        if authorized:
            
            landmark = request.pop('landmark_file',None)
            rebuild = request.pop('rebuild_file',None)
            origin = request.pop('origin_file',None)
            mask = request.pop('mask_file',None)
            stroke = request.pop('stroke_file',None)
            if landmark:
                authorize_object.landmark_file = landmark[0]
            if rebuild:
                authorize_object.rebuild_file=rebuild[0]
            if origin:
                authorize_object.origin_file= origin[0]
            if mask:
                authorize_object.mask_file= mask[0]
            if stroke:
                authorize_object.stroke_file= stroke[0]
            authorize_object.save()
            return base.Custom_Response(201,UserSerializer(authorize_object).data)
        else:
            return response
    elif request.method == 'PUT':
        authorized, authorize_object, response, request = base.Authorize_session(request)
        mode = request.get('mode')
        if authorized:
            pass
        else:
            return response
    elif request.method == 'PATCH':
        authorized, authorize_object, response, request = base.Authorize_session(request)
        mode = request.get('mode')
        if authorized:
            if mode == 'sketch_picture':
                pass

        else:
            return response
    elif request.method == 'DELETE':
        authorized, authorize_object, response, request = base.Authorize_session(request)
        if authorized:
            pass
        else:
            return response
    return base.Custom_Response(500,'not implemented passes')


@api_view(['PATCH'])
def LoginResponse(request, url=None, extra=None):
    if request.method == 'PATCH':
        mode = request.data.get('mode')
        if mode == 'login':
            authorized, authorize_object, response = base.login(
                request)

            if authorized and authorize_object != None:
                content = {'account': UserSerializer(authorize_object).data}
                return base.Custom_Response(200, content)
            else:
                return response
        elif mode == 'logout':
            base.logout(request)
            return base.Custom_Response(200, 'logout')
