#
# Run me using `. setup-environment.sh`
# Don't run me in any other way, I won't work!
#
# INSTALL CONDA
#
conda -h > /dev/null
exit_status=$?
if [ $exit_status -eq 0 ]; then
    echo "Conda installed."
else
  echo 'Conda not detected. Installing conda...'
  unameOut="$(uname -s)"
  case "${unameOut}" in
    Linux*)
        echo 'Installing conda for Linux...'
        unset PYTHONPATH
        curl -o ~/miniconda-install.sh https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
        chmod +x ~/miniconda-install.sh
        ~/miniconda-install.sh -b
        rm ~/miniconda-install.sh
        echo ". \$HOME/miniconda2/etc/profile.d/conda.sh" >> ~/.profile
        . $HOME/.profile        
        ;;
    Darwin*)
        echo 'Installing conda for MacOS...'
        curl -o ~/miniconda-install.sh https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
        chmod +x ~/miniconda-install.sh
        ~/miniconda-install.sh -b
        rm ~/miniconda-install.sh
        echo ". \$HOME/miniconda2/etc/profile.d/conda.sh" >> ~/.bash_profile
        . $HOME/.bash_profile
        ;;
    *)
        echo 'ERROR: OS not supported (only Linux and MacOS). Please install conda/miniconda manually and re-run the script.' >> /dev/stderr
        exit 1
  esac
fi
#
# INSTALL 'roboaugen' ENVIRONMENT WITH ALL DEPENDENCIES
#
ENVS=$(conda env list | awk '{print $1}' )
if [[ $ENVS = *roboaugen* ]]; then
   echo -e "Anaconda 'roboaugen' environment detected."
   conda activate roboaugen
else
   #
   # INSTALL DEPENDENCIES
   #
   echo -e "Anaconda 'roboaugen' environment not detected.\nInstalling anaconda environment 'roboaugen' together with all the required dependencies..."
   conda update --all -y
   conda update anaconda-navigator -y
   conda build purge-all
   conda env create -f environment.yml
   conda activate roboaugen
   mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
   #
   # $PYTHONPATH CONFIGURATION
   #
   touch "$CONDA_PREFIX/etc/conda/activate.d/python_path.sh"
   echo "export PYTHONPATH=$PWD" > "$CONDA_PREFIX/etc/conda/activate.d/python_path.sh"
   chmod +x "$CONDA_PREFIX/etc/conda/activate.d/python_path.sh"
   #
   # ACTIVATE CONFIGURATIONS
   #
   conda deactivate
   conda activate roboaugen
   echo "Run 'conda activate roboaugen' to enable python virtual environment to run the scripts."
fi;



