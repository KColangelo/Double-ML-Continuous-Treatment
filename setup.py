from distutils.core import setup
setup(
  name = 'ddml',
  packages = ['ddml'],
  version = '0.0.1',
  license='MIT',
  description = 'Implementation of the Double Debiased Machine Learning Estimator for Continuous Treatments',   # Give a short description about your library
  author = 'Kyle Colangelo', 
  author_email = 'kcolange@gmail.com',
  url = 'https://github.com/KColangelo/DDML',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['Machine', 'Learning', 'Double','Debiased','Causal','Inference','Nonparametric'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pytorch',
          'scikit-learn',
          'scipy',
          'numpy',
          'pandas'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Economists',
    'Topic :: Causal Inference :: Estimator',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
)