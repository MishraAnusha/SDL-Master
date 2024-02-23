from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from django.urls import path

from users import views
from .forms import LoginForm
from .views import home, profile, RegisterView, CustomLoginView, ResetPasswordView, ChangePasswordView

urlpatterns = [
                  path('', home, name='users-home'),
                  path('register/', RegisterView.as_view(), name='users-register'),
                  path('login/',
                       CustomLoginView.as_view(
                           redirect_authenticated_user=True,
                           template_name='users/login.html',
                           authentication_form=LoginForm),
                       name='login'
                       ),

                  path('logout/', auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),

                  path('password-reset/', ResetPasswordView.as_view(), name='password_reset'),

                  path('password-reset-confirm/<uidb64>/<token>/',
                       auth_views.PasswordResetConfirmView.as_view(
                           template_name='users/password_reset_confirm.html'),
                       name='password_reset_confirm'),

                  path('password-reset-complete/',
                       auth_views.PasswordResetCompleteView.as_view(
                           template_name='users/password_reset_complete.html'),
                       name='password_reset_complete'),

                  path('password-change/', ChangePasswordView.as_view(), name='password_change'),
                  path('profile/', profile, name='users-profile'),
                  path('activate/<uidb64>/<token>', views.activate, name='activate'),
                  path('get_all_users/', views.get_all_users, name='get_all_users'),
                  path('H_Control/', views.H_Control, name='H_Control'),
                path('store-hwfeeds/', views.store_hwfeeds, name='store_hwfeeds'),
                path('process_parameters/', views.process_parameters, name='process_parameters')
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
