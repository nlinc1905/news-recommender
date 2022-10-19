from scipy.stats import beta
from django.db import models


class Campaign(models.Model):
    """Data model for AB test campaign"""
    campaign_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(
        blank=True,
        default='',
        help_text='Description of the AB test'
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=True)
    allow_repeat_impression_by_same_user = models.BooleanField(
        default=True,
        help_text='True if repeat impressions/conversions allowed by the same user'
    )

    def __str__(self):
        return self.name


class Variant(models.Model):
    """Data model for variants (treatments) in an AB test campaign"""
    campaign_id = models.ForeignKey(Campaign, on_delete=models.CASCADE)
    variant_code = models.CharField(max_length=32)
    name = models.CharField(max_length=255)
    html_template = models.FilePathField(null=True)
    impressions = models.IntegerField(
        default=1,
        help_text='Number of times variant was shown/visited'
    )
    conversions = models.IntegerField(
        default=1,
        help_text='Number of conversions for variant'
    )
    conversion_rate = models.FloatField(
        default=1.0,
        help_text='conversions / impressions'
    )

    def beta_pdf(self, x_vals):
        """
        Get the beta distribution (conjugate prior) values, given corresponding
        x values where < x < 1.  Beta distribution's alpha = conversions and
        beta = impressions - conversions.
        """
        y_vals = list(beta.pdf(
            x_vals,
            max(self.conversions, 1),
            max(self.impressions-self.conversions, 1)
            )
        )
        return y_vals

    def __str__(self):
        return f'{self.campaign_code}:{self.variant_code}'
