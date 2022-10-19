from abtest.models import Campaign, Variant


# Set up an AB Testing campaign
campaign, _ = Campaign.objects.get_or_create(
    name="Test Homepage",
    description="Testing Homepage designs"
)

# create variants
for code in ['a', 'b']:
    variant, _ = Variant.objects.get_or_create(
        campaign_id=campaign,
        variant_code=code,
        name=f'Homepage {code}',
        html_template=f'abtest/home_variant_{code}.html'
    )
