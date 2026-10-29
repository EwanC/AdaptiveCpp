set pagination off
set confirm off
set print thread-events off
set debuginfod enabled off
handle SIGUSR1 stop print nopass
handle SIGPIPE nostop noprint pass
run
while 1
  thread apply all bt 30
  continue
end
