    #!/bin/bash
    while true; do
        vcgencmd measure_temp
        sleep 60 # Sleep for 60 seconds (1 minute)
    done
