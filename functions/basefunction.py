from reface_main.models import *
from functions.dependency_imports import *
from functions.non_dependency import *
def Custom_Response(status, string=''):
    if status == 200:
        status = Status.HTTP_200_OK
    elif status == 102:
        string = string + 'processing'
        status = Status.HTTP_200_OK
    elif status == 100:
        string['info'] = 'waitng for '+string.get('info') + ' request'
        status = Status.HTTP_100_CONTINUE
    elif status == 201:
        status = Status.HTTP_201_CREATED
    elif status == 202:
        status = Status.HTTP_202_ACCEPTED
    elif status == 404:
        string = string+'does not exist'
        status = Status.HTTP_404_NOT_FOUND
    elif status == 401:
        string = 'not authorized'+string
        status = Status.HTTP_401_UNAUTHORIZED
    elif status == 403:
        string = 'forbidden'+string
        status = Status.HTTP_403_FORBIDDEN
    elif status == 406:
        string = string
        status = Status.HTTP_406_NOT_ACCEPTABLE
    elif status == 409:
        string = "existing "+string
        status = Status.HTTP_409_CONFLICT
    elif status == 412:
        status = Status.HTTP_412_PRECONDITION_FAILED
    elif status == 500:
        string = 'something wrong to server manager'
        status = Status.HTTP_500_INTERNAL_SERVER_ERROR
    elif status == 502:
        status = Status.HTTP_501_NOT_IMPLEMENTED

    error_object = {'info': string, 'query_count': len(connection.queries)}
    reset_queries()

    return Response(error_object, status=status)

def Findaccount(caccount):  # really find account object
    try:
        return User.objects.get(account=caccount)
    except User.DoesNotExist:
        return None


def logout(request):
    request.session.clear()


def login(request):
    authorized = 0
    authorize_object = None
    response = None
    if request.method == 'POST' or request.method == 'DELETE':
        request_method = getattr(request, 'POST')
    elif request.method == 'GET':
        request_method = getattr(request, 'GET')
    else:
        request_method = request.data

    if set(('account', 'password')) <= request_method.keys():
        account = request_method.get('account').lower()
        authorize_object = Findaccount(account)
        if authorize_object and authorize_object.check_password(request_method.get('password')):
            request.session['account'] = account
            request.session['password'] = request_method.get(
                'password')
            authorized = 1
    if authorized != 1:
        response = Custom_Response(401, 'id/password')

    return authorized, authorize_object, response


def Authorize_session(request,restriction=3, specific=[-1]):
    authorized = 0
    authorize_object = None
    response = None
    if request.method == 'POST' or request.method == 'DELETE':
        request_method = getattr(request, 'POST')
    elif request.method == 'GET':
        request_method = getattr(request, 'GET')
    else:
        request_method = request.data

    if 'account' in request.session and 'password' in request.session:
        account = request.session['account'].lower()
        password = request.session['password']
        authorize_object = Findaccount(account)
        if authorize_object.check_password(password):
            authorized = 1

    if authorized != 1:
        response = Custom_Response(401, 'login required')


    request = gen_request_code(request_method.copy(), request.FILES)
    return authorized, authorize_object, response, request
