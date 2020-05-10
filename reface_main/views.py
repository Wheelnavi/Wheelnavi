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
                user_code = request.get('user_code')

                onetake(user_code,image_name,dbface=True,readdat=True)
                return base.Custom_Response(502,'not implemented')
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
            newuser = UserSerializer(request)
            if newuser.is_valid():
                newuserobj = newuser.save()
                if landmark:
                    newuserobj.landmark_file = landmark[0]
                if rebuild:
                    newuserobj.rebuild_file=rebuild[0]
                if origin:
                    newuserobj.origin_file= origin[0]
                if mask:
                    newuserobj.mask_file= mask[0]
                if stroke:
                    newuserobj.stroke_file= stroke[0]
                newuserobj.save()
                return base.Custom_Response(201,newuser.data)
            else:
                return base.Custom_Response(406,newuser.errors)
        else:
            return response
    elif request.method == 'PUT':
        authorized, authorize_object, response, request = base.Authorize_session(request)
        if authorized:
            pass
        else:
            return response
    elif request.method == 'PATCH':
        authorized, authorize_object, response, request = base.Authorize_session(request)
        if authorized:
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
