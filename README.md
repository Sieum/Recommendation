# Recommendation

### Django 설치 가이드

> $ pip3 install pipenv   // 가상환경 설치
>
>$ pipenv shell    // 가상환경 실행
>
>$ pipenv install django==4.1   // 장고 설치 (반드시 4.1 이하버젼 깔아야함, 4.2이상부터는 mysql 버젼 8 이상만 지원
> 
>$ pip freeze   // 장고가 설치되었는지 확인하는 명령어
> 
>$ pip install mysqlclient   // 파이썬용 mysql connector 설치
> 
>$ pip install djangorestframework   // 장고 rest api 사용하기 위해 drf 라고 부르는 djangorestframework 설치
> 
>$ pip install django-cors-headers   // 장고 cors 에러 해결을 위해 설치
> 
>$ django-admin startproject "프로젝트 명"    // 프로젝트 생성
> 
>$ python manage.py startapp "앱 명"   // 앱 생성, 스프링의 패키지처럼 장고는 하나의 프로젝트 안에 여러개의 앱 구조로 이루어져있다.
> 
>$ python manage.py runserver  // 장고 실행 방법
