mkdir -p uploads
mkdir -p processes
#python worker.py &>>worker1.log &
#python worker.py &>>worker2.log  &
#python worker.py &>>worker3.log &
python supervisor.py &>>supervisor.log &
python worker.py &>>worker3.log &
python server.py  
