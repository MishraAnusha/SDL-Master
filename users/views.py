from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMessage
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.utils.encoding import force_str, force_bytes
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.views import View
import json
from django.shortcuts import render
from django.http import JsonResponse

from .forms import RegisterForm, LoginForm, UpdateUserForm, UpdateProfileForm
from .token import account_activation_token


def home(request):
    return render(request, 'users/home.html')


class RegisterView(View):
    form_class = RegisterForm
    initial = {'key': 'value'}
    template_name = 'users/register.html'

    def dispatch(self, request, *args, **kwargs):
        # will redirect to the home page if a user tries to access the register page while logged in
        if request.user.is_authenticated:
            return redirect(to='/')

        # else process dispatch as it otherwise normally would
        return super(RegisterView, self).dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)

        if form.is_valid():
            # form.save()
            user = form.save(commit=False)
            user.is_active = False
            user.save()

            username = form.cleaned_data.get('username')
            # email verification
            to_list = ['iotproject213@gmail.com'] if settings.DEBUG else [user.email]
            current_site = get_current_site(request)
            email_subject = 'Activate your account'
            msg = render_to_string('emails/acc_active_email.html', {
                'name': user.first_name,
                'domain': current_site.domain,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': account_activation_token.make_token(user),
            })

            email = EmailMessage(email_subject, msg,
                                 settings.EMAIL_HOST_USER, to=to_list)

            email.fail_silently = True
            email.send()
            messages.success(
                request, f'Account created for {username}, We have sent you an email to verify your account')

            return redirect(to='login')

        return render(request, self.template_name, {'form': form})


# Class based view that extends from the built-in login view to add remember me functionality
class CustomLoginView(LoginView):
    form_class = LoginForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')

        if not remember_me:
            # set session expiry to 0 seconds. So it will automatically close the session after the browser is closed.
            self.request.session.set_expiry(0)

            # Set session as modified to force data updates/cookie to be saved.
            self.request.session.modified = True

        # else browser session will be as long as the session cookie time "SESSION_COOKIE_AGE" defined in settings.py
        return super(CustomLoginView, self).form_valid(form)


class ResetPasswordView(SuccessMessageMixin, PasswordResetView):
    template_name = 'users/password_reset.html'
    email_template_name = 'users/password_reset_email.html'
    subject_template_name = 'users/password_reset_subject'
    success_message = "We've emailed you instructions for setting your password, " \
                      "if an account exists with the email you entered. You should receive them shortly." \
                      " If you don't receive an email, " \
                      "please make sure you've entered the address you registered with, and check your spam folder."
    success_url = reverse_lazy('users-home')


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'users/change_password.html'
    success_message = "Successfully Changed Your Password"
    success_url = reverse_lazy('users-home')


@login_required
def profile(request):
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)
        profile_form = UpdateProfileForm(
            request.POST, request.FILES, instance=request.user.profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile is updated successfully')
            return redirect(to='users-profile')
    else:
        user_form = UpdateUserForm(instance=request.user)
        profile_form = UpdateProfileForm(instance=request.user.profile)

    return render(request, 'users/profile.html', {'user_form': user_form, 'profile_form': profile_form})


def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except(TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()
        messages.success(request, 'Your account has been verified')
        return redirect(to='login')
    else:
        messages.error(request, 'The link is invalid')
        return redirect(to='login')


def get_all_users(request):
    users = User.objects.all()
    return render(request, 'users/all_users.html', {'users': users})

def H_control(request):
    return render(request, 'H_control/MVP.html')
    
def Feeds(request):
    return render(request, 'nodes/feeds.html')

def process_parameters(request):
    if request.method == 'POST':
        node_id=request.POST.get('node_id','')
        mvp = request.POST.get('mvp', '')
        mvs = request.POST.get('mvs', '')
        svp = request.POST.get('svp', '')
        svs = request.POST.get('svs', '')
        ro_1 = request.POST.get('ro_1', '')
        ro_2 = request.POST.get('ro_2', '')
        
        
        # Create a dictionary to store form data
        form_data = {
            'node_id':node_id,
            'mvp': mvp,
            'mvs': mvs,
            'svp': svp,
            'svs': svs,
            'ro_1': ro_1,
            'ro_2': ro_2
            
        }
        
        # Return form data in JSON response
        return JsonResponse({'success': True, 'form_data': form_data})
    else:
        return JsonResponse({'error': 'Invalid request method'})

from .models import Feeds
def store_hwfeeds(request):
    if request.method == "POST":
        # Store data to the database
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        # Assuming 'node_id' is part of the request body
        node_id = body.get('node_id')
        if node_id:
            # Assuming Feeds model has fields MVP, MVS, SVP, SVS, RO_1, RO_2
            feed = Feeds.objects.create(
                node_id=node_id,
                MVP=body.get('MVP'),
                MVS=body.get('MVS'),
                SVP=body.get('SVP'),
                SVS=body.get('SVS'),
                RO_1=body.get('RO_1'),
                RO_2=body.get('RO_2'),
            )
            feed.save()
            return JsonResponse({'message': 'Data stored successfully'})
        else:
            return JsonResponse({'error': 'Node ID not provided'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
