from django.urls import path

from nodes import views

urlpatterns = [
    path('', views.node_list, name='nodes'),
    path('store_feeds', views.store_feeds, name='store_feeds'),
    path('get_feeds_chart/<int:node_id>', views.get_feeds, name='get_feeds'),
    path('get_feeds_table/<int:node_id>',
         views.get_feeds_table, name='get_feeds_table'),
    path('crud/', views.CrudNodes.as_view()),
    path('delete_node/<int:node_id>', views.delete_node, name='delete_node'),
    path('get_chart/<int:node_id>', views.get_chart_data, name='get_chart'),
    path('upload_image/<int:node_id>',
         views.crop_image_upload, name='upload_image'),
    path('crop_image_gallery/<int:node_id>',
         views.crop_image_gallery, name='crop_image_gallery'),
    path('upload_image/<int:node_id>',
         views.crop_image_upload, name='upload_image'),
    path('crop_image_gallery/<int:node_id>',
         views.crop_image_gallery, name='crop_image_gallery'),
    path('import-csv/<int:node_id>', views.import_csv, name='import_csv'),
    path('export_feeds_csv/<int:node_id>/',
         views.export_feeds_csv, name='export_feeds_csv'),

    # particular User Nodes
    path('user_nodes/<int:user_id>', views.node_particuler_list, name='user_nodes'),
    path('user_nodes/store_feeds', views.store_feeds, name='user_store_feeds'),
    path('user_nodes/get_feeds_chart/<int:node_id>',
         views.get_feeds, name='user_get_feeds'),
    path('user_nodes/get_feeds_table/<int:node_id>',
         views.get_feeds_table, name='user_get_feeds_table'),
    path('user_nodes/crud/', views.CrudNodes.as_view()),
    path('user_nodes/delete_node/<int:node_id>',
         views.delete_node, name='user_delete_node'),
    path('user_nodes/get_chart/<int:node_id>',
         views.get_chart_data, name='user_get_chart'),
    path('user_nodes/upload_image/<int:node_id>',
         views.crop_image_upload, name='user_upload_image'),
    path('user_nodes/crop_image_gallery/<int:node_id>',
         views.crop_image_gallery, name='user_crop_image_gallery'),
    path('user_nodes/import-csv/<int:node_id>',
         views.import_csv, name='user_import_csv'),
    path('user_nodes/export_feeds_csv/<int:node_id>/',
         views.export_feeds_csv, name='user_export_feeds_csv'),


]
