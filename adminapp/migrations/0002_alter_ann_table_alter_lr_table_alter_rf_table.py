# Generated by Django 4.0 on 2024-10-15 09:15

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('adminapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='ann',
            table='ANN',
        ),
        migrations.AlterModelTable(
            name='lr',
            table='LR',
        ),
        migrations.AlterModelTable(
            name='rf',
            table='RF',
        ),
    ]
