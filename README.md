Steps to get the app running using Docker:

1. Install Docker (shocking, I know). I recommend using the Docker Desktop GUI app to build this container, as well as the built-in terminal that comes with it.

2. Open up that terminal and run command:
```bash
  docker pull espimkii/ml-demo-flask-app:2.0
```
   This pulls the image built from the contents of this repo to your local Docker image list.

3. Check the Images menu on the left side of your Docker Desktop GUI. It should have the image you just pulled.
   Run it using the "Run" button under "Actions"

4. In the "Optional Settings" prompt, put 5000 as the Host port, then Run.

5. The container should automatically start a localhost Flask app on your computer.
   Punch in your web browser:
```
  http://127.0.0.1:5000
```
   And use the app.
