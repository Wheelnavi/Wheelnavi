from functions.dependency_imports import *


def gen_request_code(request, extra=None):
    request_code = random.choices(
        string.ascii_letters + string.digits, k=7)
    request['request_code'] = request_code
    request.update(extra)
    return request
