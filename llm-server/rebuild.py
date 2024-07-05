import subprocess

def main():
    process = subprocess.Popen(['docker', 'ps', '-a'], stdout=subprocess.PIPE)
    stdout, _ = process.communicate()
    output = stdout.decode('utf-8')
    for line in output.split('\n'):
        if 'llm_server_llamadoc' in line:
            container_id = line.split()[0]
            break
    else:
        container_id = None

    if container_id is not None:
        process = subprocess.Popen(['docker', 'stop', container_id], stdout=subprocess.PIPE)
        stdout, _ = process.communicate()
        print(stdout.decode('utf-8'))

    process = subprocess.Popen(['git', 'pull'], stdout=subprocess.PIPE)
    stdout, _ = process.communicate()
    print(stdout.decode('utf-8'))

    process = subprocess.Popen(['sh', 'build.sh'], stdout=subprocess.PIPE)
    stdout, _ = process.communicate()
    print(stdout.decode('utf-8'))
    

if __name__ == '__main__':
    main()
    