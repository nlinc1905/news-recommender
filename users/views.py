from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.generic import CreateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.http import HttpResponseRedirect

from .forms import (
    CustomUserCreationForm,
    DeleteUserForm,
)


class SignupPageView(CreateView):
    """
    View for user sign up
    """
    form_class = CustomUserCreationForm
    template_name = 'users/signup.html'

    def form_valid(self, form):
        """
        Specify custom actions to take on form validation.
        This will save the user, get the saved username
        and password, authenticate, then log the user in
        automatically.  This prevents the user from having
        to sign up, and then login.  After signing up, the
        user will be logged in and redirected.
        """
        if self.request.user.is_authenticated:
            return redirect('home')
        else:
            # when form is valid, save the user to the DB
            # this creates the user
            form.save()
            username = self.request.POST['username']
            password = self.request.POST['password1']
            # authenticate user, then login
            user = authenticate(username=username, password=password)
            login(self.request, user)
            return HttpResponseRedirect(reverse('home'))


@login_required(login_url='login')
def delete_user_view(request):
    """
    View to delete user
    """
    if request.method == 'POST':
        form = DeleteUserForm(data=request.POST, instance=request.user)
        if form.is_valid():
            # delete user
            request.user.delete()
            return redirect('home')
    else:
        form = DeleteUserForm(instance=request.user)
        args = {'form': form}
        return render(request, 'users/delete_user.html', args)
