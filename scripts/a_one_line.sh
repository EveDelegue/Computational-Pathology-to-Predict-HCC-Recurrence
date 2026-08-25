folder_name="/mnt/backup/aziz_chaari/asadraoui_data_backups/backup_HM/Patient_160"
dest_folder="data/WSIs/HM"
mkdir $dest_folder
cp -r $folder_name $dest_folder
echo "process"
rm -r $dest_folder