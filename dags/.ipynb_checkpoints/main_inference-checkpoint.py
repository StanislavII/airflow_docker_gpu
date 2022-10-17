{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ad89edf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n",
    "# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT \n",
    "# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS \n",
    "# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE \n",
    "# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, \n",
    "# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, \n",
    "# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; \n",
    "# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER \n",
    "# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT \n",
    "# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN \n",
    "# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE \n",
    "# POSSIBILITY OF SUCH DAMAGE.\n",
    "\n",
    "\n",
    "from datetime import datetime\n",
    "\n",
    "from airflow import DAG\n",
    "from airflow.operators.dummy import DummyOperator\n",
    "from airflow.decorators import task\n",
    "from airflow.utils.edgemodifier import Label\n",
    "\n",
    "# Docker library from PIP\n",
    "import docker\n",
    "\n",
    "# Simple DAG\n",
    "with DAG(\n",
    "    \"test_inference\", \n",
    "    schedule_interval=\"@daily\", \n",
    "    start_date=datetime(2022, 1, 1), \n",
    "    catchup=False, \n",
    "    tags=['test']\n",
    ") as dag:\n",
    "\n",
    "\n",
    "    @task(task_id='run_translation')\n",
    "    def run_gpu_translation(**kwargs):\n",
    "\n",
    "        # get the docker params from the environment\n",
    "        client = docker.from_env()\n",
    "          \n",
    "            \n",
    "        # run the container\n",
    "        response = client.containers.run(\n",
    "\n",
    "             # The container you wish to call\n",
    "             'inference:latest',\n",
    "\n",
    "             # The command to run inside the container\n",
    "             'python3 main_inference.py',\n",
    "\n",
    "             # Passing the GPU access\n",
    "             device_requests=[\n",
    "                 docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])\n",
    "             ], \n",
    "             \n",
    "             # Give the proper system volume mount point\n",
    "             volumes=[\n",
    "                 '<HOST_DATA_FOLDER>:/data'\n",
    "             ]\n",
    "        )\n",
    "\n",
    "        return str(response)\n",
    "\n",
    "    run_translation = run_gpu_translation()\n",
    "\n",
    "\n",
    "    # Dummy functions\n",
    "    start = DummyOperator(task_id='start')\n",
    "    end   = DummyOperator(task_id='end')\n",
    "\n",
    "\n",
    "    # Create a simple workflow\n",
    "    start >> run_translation >> end"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}