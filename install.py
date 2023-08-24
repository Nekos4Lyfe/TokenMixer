
def install_tokenmixer_dependencies():

  import os
  import pkg_resources
  from functools import lru_cache
  from rich import print_json
  index_url = os.environ.get('INDEX_URL', "")
  import subprocess
  import sys
  req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

  # check if package is installed
  def installed(package, friendly: str = None):
    ok = True
    try:
        if friendly:
            pkgs = friendly.split()
        else:
            pkgs = [p for p in package.split() if not p.startswith('-') and not p.startswith('=')]
            pkgs = [p.split('/')[-1] for p in pkgs] # get only package name if installing from url
        for pkg in pkgs:
            if '>=' in pkg:
                p = pkg.split('>=')
            else:
                p = pkg.split('==')
            spec = pkg_resources.working_set.by_key.get(p[0], None) # more reliable than importlib
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(p[0].lower(), None) # check name variations
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(p[0].replace('_', '-'), None) # check name variations
            ok = ok and spec is not None
            if ok:
                version = pkg_resources.get_distribution(p[0]).version
                # log.debug(f"Package version found: {p[0]} {version}")
                if len(p) > 1:
                    exact = version == p[1]
                    ok = ok and (exact)
                    if not exact:
                      print(f"Package wrong version: {p[0]} {version} required {p[1]}")
            else:
                print(f"Package version not found: {p[0]}")
        return ok
    except ModuleNotFoundError:
        print(f"Package not installed: {pkgs}")
        return False


  @lru_cache()
  def is_installed(package): # compatbility function
    return installed(package)


  @lru_cache()
  def run(command, desc=None, errdesc=None, custom_env=None, live=False): # compatbility function
    if desc is not None:
        print(desc)
    if live:
        result = subprocess.run(command, check=False, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'} Command: {command} Error code: {result.returncode}""")
        return ''
    result = subprocess.run(command, stdout=subprocess.PIPE, check=False, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)
    if result.returncode != 0:
        raise RuntimeError(f"""{errdesc or 'Error running command'}: {command} code: {result.returncode}
  {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else ''}
  {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else ''}
  """)
    return result.stdout.decode(encoding="utf8", errors="ignore")


  @lru_cache()
  def run_pip(pkg, desc=None): # compatbility function
    if desc is None:
        desc = pkg
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{sys.executable}" -m pip {pkg} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


  with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                package_name, package_version = package.split('==')
                installed_version = pkg_resources.get_distribution(package_name).version
                if installed_version != package_version:
                    run_pip(f"install {package}", f"TokenMixer requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif not is_installed(package):
                run_pip(f"install {package}", f"TokenMixer requirement: {package}")
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {package}, some preprocessors may not work.')
##############

install_tokenmixer_dependencies()
